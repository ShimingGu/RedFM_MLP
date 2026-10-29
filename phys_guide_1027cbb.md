# Stage 2B.6 — Adaptive Physical-Likelihood Trust and Matched-Convergence Architecture Tests

## Scientific objective

Determine whether a physics-aware neural model can exploit reliable SED forward-model information while remaining robust to physical-model misspecification.

The scientific question is:

**Can a neural model use a physically informative redshift likelihood when justified by the observations, and avoid being dominated by misleading physical scores when the SED model is inadequate?**

A second independent question is whether the previous N2 advantage over N1 persists after sufficiently comparable source convergence.

This experiment tests numerical and physical inductive bias.

It does not test pretrained Qwen knowledge.

No Qwen backbone or LLM training is permitted.

## 1. Scientific background

Read the reports and experimental manifests from Stage 2A.5, Stage 2B.5 and Stage 2A.6.

Previous experiments established:

- N1 is a shared mask-aware band encoder with a neural redshift-PDF head.
- N2 adds a nonnegative SED coefficient predictor and a physical profile-likelihood score.
- N2 uses the verified full filter-response integration kernel.
- The original combination is:

`logits = 4*tanh(raw/4) + 0.2*(-0.5*(chi2-min_chi2)/Nvalid)`

- The resulting physical-score temperature is Nvalid/0.2, equal to 25 for five valid bands and 55 for eleven.
- The existing 10% fractional template-error term further broadens the physical likelihood.
- Approximately 80.8% of source synthetic bands had TEF variance larger than reported measurement variance.
- The original synthetic identifiable-source physical-only score had 50.0% outliers.
- Removing score tempering, with frozen weights, reduced that identifiable-source figure to 0.59%.
- In mismatched SED regimes, increasing physics weight could degrade combined-model results.
- In a held-out identifiable full-response synthetic test, N2 reduced catastrophic outliers from 42.97% for N1 to 30.08%.
- The paired 256-SED bootstrap interval supported this limited difference.
- The strictly new-only-filter interval crossed zero.
- The independent physical NNLS model remained considerably stronger.
- N1/N2 were not fully converged at the original training cutoff.
- Longer source-only convergence diagnostics were performed but not evaluated on held-out responses.

Stage 2A.6 separately investigates how SED dimensionality, coefficient priors, marginalisation and bandpass coverage affect the physical likelihood.

Use its verified infrastructure and findings without assuming that any newly constrained likelihood is universally correct.

## 2. Hard constraints

- No Qwen, QLoRA, DoRA, LLM training or Stage 2C.
- No CLAUDS ZPHOT for model training, model selection, calibration or iterative experimental decisions.
- No held-out real-survey test data.
- Do not overwrite previous synthetic truth sets, checkpoints or reports.
- Preserve source-training/source-validation identity separation.
- Preserve all actual response curves, masks and physical units.
- Preserve negative flux measurements.
- Do not infer missing EBV values.
- Do not insert true redshift, true SED identity, known mixture coefficients or model-mismatch labels into deployable inference features.
- Keep oracle-only conditions clearly separated from deployable models.
- Do not call a wider or less concentrated PDF automatically more physically correct.
- Use matched supervision and information budgets wherever possible.

Stage 2B.6 is a new controlled experiment, not a retrospective re-ranking of the registered Stage 2B pilot.

## 3. Physical likelihood candidates

Use the Stage 2A.6 handoff to identify a small number of physically justified scores.

At minimum retain:

**P-original — Original 12-template physics**

The previous full-basis profile score, with its documented limitations.

**P-constrained — Constrained SED model**

A source-validated low-dimensional SED score from Stage 2A.6, if one is numerically and physically defensible.

**P-marginal — Marginalised SED evidence**

Include this only if Stage 2A.6 established a numerically convergent and properly normalised calculation.

Do not use a marginal score merely because it produces narrower posteriors or lower source CE.

For each candidate, document:

- Model class and SED freedom.
- Filter-response integration.
- Template or coefficient prior.
- Amplitude treatment.
- Observational likelihood.
- TEF or model-discrepancy prescription.
- Numerical normalization.
- Computational cost.
- Known failure modes.

An original and a constrained score may both be retained as diagnostics even if Stage 2A.6 finds no clear winner.

## 4. Build controlled matched and mismatched physics datasets

Construct a fixed source-only synthetic experimental suite.

### Domain M1 — Identifiable matched physics

Generate spectra from the exact constrained SED family assumed by the physical likelihood.

Use a limited, identifiable mixture distribution.

Include actual HSC/Legacy responses and moderate noise.

Verify identifiability independently using a physical oracle that does not supply its privileged latent information to the neural models.

### Domain M2 — Underdetermined matched physics

Generate arbitrary mixtures of the larger SED basis.

Use the same physical basis as the decoder, but allow enough latent freedom that five-band photometry can become non-identifiable.

Distinguish model correctness from identifiability.

Even a mathematically correct likelihood may produce multiple redshift modes.

### Domain M3 — Deliberately mismatched physics

Generate spectra from analytic/CWW families or other held-out physically motivated SED classes not fully represented by the decoder's template space.

Include continuum-slope, spectral-break and emission-feature variations.

The neural and physical models must encounter identical observations.

### Domain M4 — Controlled measurement mismatch

Use a limited set of predeclared perturbations such as:

- Small zero-point or wavelength-dependent flux discrepancies.
- Unmodelled dust differences.
- Moderate changes to the measurement-noise model.
- Missing-band scenarios.
- Correct versus perturbed response curves.

Keep these distinct from intrinsic SED mismatch.

Do not claim that arbitrary perturbations reproduce actual HSC/CLAUDS calibration history.

### Source/held-out partitions

Split by intrinsic SED identity before generating redshifts, response systems and noise realizations.

Use source-filter observations for training and source validation.

Reserve independent SED identities for held-out filter transfer and held-out SED-family evaluation.

Use predetermined sampling seeds and adequate numbers of unique SEDs for paired uncertainty estimates.

Avoid relying exclusively on previously inspected Stage 1/2B.5 held-out objects for new confirmation.

## 5. Matched-convergence N1 versus N2 experiment

This is a mandatory component.

Implement:

### N1 — Neural-only shared-band encoder

Shared per-band numerical encoder, mask-aware aggregation and redshift-PDF head.

### N2 — Same encoder and head plus physical decoder

Match N1's encoder and neural head architecture.

Add the physical decoder and distinguish its extra parameters explicitly.

Where appropriate, test both the original physical score and the Stage 2A.6 constrained score.

All models must receive the same observed fluxes, uncertainties, masks and filter information.

Do not give N2 hidden knowledge of true SED coefficients or redshifts at inference.

### Training protocol

Use identical source-training and validation SED identities.

Use comparable optimizer settings, random seeds, label counts and total supervised exposure.

Run paired seeds rather than only one initialization.

For N1 and N2, report comparisons at:

1. The same number of optimizer updates.
2. Similar source-validation convergence.
3. A fixed maximum compute budget.

Convergence should be assessed using predeclared source-validation criteria and actual loss curves, not by choosing whichever stopping point favours N2.

Track:

- Training CE.
- Source-validation CE.
- Gradient norms.
- Training/validation divergence.
- PDF entropy.
- Prediction concentration.
- Source point-estimate metrics.
- N2 reconstruction loss and coefficient behaviour.

An architecture that has not converged cannot be ranked fairly against one that has.

If convergence cannot be established within reasonable resources, retain the comparison as inconclusive.

Do not use held-out-filter performance to choose the number of updates.

## 6. Fixed physical-score trust controls

First investigate physics weighting without adding a learned gate.

For identical frozen N2 models and the same source-validation objects, compare:

- Neural-only.
- Physical-only with original tempering.
- Physical-only without original tempering.
- Original N2 combination.
- A small predetermined set of intermediate physical-score weights.
- Stage 2A.6 constrained physics under equivalent conventions.

Separate:

1. TEF or observational-noise assumptions.
2. Physical-score temperature.
3. Neural-logit scale.
4. The choice of SED model.

Do not vary all these factors simultaneously and then attribute improvement to one.

For example, compare physics scores using the same measurement likelihood while varying only the temperature.

Then compare error models at the same temperature.

Preserve both point estimates and proper probabilistic scores.

Evaluate whether increasing physics trust helps matched spectra while hurting mismatched spectra.

A strong score under exact generative assumptions is an expected positive control, not sufficient evidence of realistic cross-survey generalisation.

## 7. Adaptive physical-score trust

If the fixed-score experiments demonstrate meaningful differences between reliable and unreliable physical regimes, implement one compact trust mechanism.

Schematic form:

`p(z | D) = softmax(neural_logits(z) + g(D) * physical_score(z))`

where:

`g(D) >= 0`

and D contains only observed photometry, uncertainties, masks, response information and source-validated model-adequacy diagnostics.

Consider simple low-capacity alternatives before a flexible gating network.

Possible trust features include:

- Number of valid bands.
- Photometric S/N and uncertainty structure.
- Source-defined input support diagnostics.
- Width and multimodality of the physical redshift likelihood.
- Agreement or disagreement between independently defined physical models.
- Physical reconstruction residuals with documented degrees-of-freedom limitations.
- Cross-model predictive disagreement.
- Sensitivity of the physical likelihood to small justified changes in template assumptions.

Important limitations:

- A minimum chi-square near zero does not imply correct physics when the SED mixture is underdetermined.
- Broad likelihoods are not automatically unreliable.
- Narrow likelihoods are not automatically correct.
- Model disagreement can signal misspecification but is not a calibrated failure probability.
- A gate that learns to ignore physics on every difficult galaxy is not a demonstrated physical improvement.

### Suggested initial trust models

Compare:

**G0 — Fixed global coefficient**

The existing N2 weighting.

**G1 — Simple source-defined rule**

A low-capacity bounded weighting rule using predeclared adequacy indicators.

**G2 — Small learned gate**

Only if G1 and the fixed controls justify additional flexibility.

The learned gate should have very few trainable parameters.

Train it using source-only labels and source-defined train/validation splits.

Do not give it the true simulated model-mismatch class at inference.

A mismatch-class oracle may be used as a labelled upper bound, but never included as a deployable input or cited as realistic model performance.

Regularise the gate and report whether it collapses to always-on or always-off physics.

The trust mechanism should be tested against a fixed physics-only baseline and against N1.

## 8. Distinguish posterior weighting from redshift identifiability

For every physical score, inspect:

- True-redshift profile/evidence values.
- Incorrect competing redshift modes.
- PDF calibration.
- Confidence assigned to incorrect solutions.
- Model residuals at true and inferred redshifts.

A temperature parameter changes the relative posterior weighting of redshift bins, but does not generally fix the location of incorrect chi-square minima.

A source-calibrated probability distribution is not necessarily physically correct on an unseen SED family.

The experiment must distinguish:

- Better numerical calibration.
- Better redshift discrimination.
- Better resilience to SED mismatch.
- Better transfer to unseen filters.

Do not combine them into one undefined success criterion.

## 9. Mandatory independent baselines

Keep:

- N0 conventional numerical MLP as a secondary numerical baseline.
- Properly trained N1 shared-band encoder.
- N2 with original physics.
- N2 with the constrained physical score.
- Physical-only likelihood with the same available information.
- Stage 2A.6 profile/marginal physical baselines where compatible.

When the physical baseline is allowed to fit arbitrary SED coefficients at every redshift while N2 predicts only one global SED mixture, document this difference.

Likewise, if N2 receives a complete external physical-response kernel and N1 uses a compressed response representation, the comparison must not be described as identical-information physics attribution.

Include a targeted full-response-information control where feasible.

The principal architecture comparison remains N1 versus N2 under matched source supervision and appropriately documented information access.

## 10. Held-response transfer

Only evaluate frozen, source-selected models on the new held-out sets after the source-learning and correctness gates pass.

For each matched/mismatched physics domain report:

- Source-system validation.
- Held-out CLAUDS-like responses.
- Strict new-only filter responses.
- Held-out SED-family tests.
- Missing-band/noise stress tests.
- Physically regenerated response counterfactuals.

Paired observations must preserve intrinsic SED/redshift identity while changing the observational system.

Include a known-redshift-change test in which the same intrinsic SED is reobserved at a different redshift.

Do not interpret a response to incorrect metadata as proof of physical reasoning.

Report:

- NMAD.
- Catastrophic-outlier fraction.
- Bias.
- CE and CRPS.
- 68% coverage.
- PIT/calibration.
- Prediction concentration.
- Cross-filter paired differences.
- Redshift-response accuracy.
- Physical reconstruction adequacy.
- Outlier changes relative to N1.
- Computational cost.

Use paired SED-level bootstrap.

Also assess variation across independently trained seeds.

Distinguish training-seed variability from conditional test-SED sampling uncertainty.

## 11. Source-only real-data plausibility check

If synthetic correctness and source-convergence gates pass, perform a limited real-source diagnostic with existing audited HSC/Legacy paired galaxies.

Do not initiate population-scale training.

Do not evaluate CLAUDS target labels.

Compare the physical trust diagnostics between:

- HSC grizy.
- Legacy grz.
- Legacy grzW1W2.
- Available audited HSC+WISE combinations.

Use actual source spectroscopic redshifts only for evaluation, not for deployable trust-feature calculation.

Determine whether the gate tends to trust underdetermined HSC likelihoods merely because their minimum chi-square is near zero.

Do not declare the synthetic gate validated on real galaxies simply because it has reasonable numerical behaviour.

If adequate real-source high-z coverage is absent, document that limitation explicitly.

## 12. Decision gates

### Gate 1 — Physical correctness

Verify kernels, likelihoods, masking, nuisance-amplitude profiling, numerical stability and gradients.

### Gate 2 — Source convergence

N1 and N2 must demonstrate meaningful source-validation learning.

Matched-convergence comparison must be possible or explicitly declared inconclusive.

### Gate 3 — Conditional physical benefit

Test whether N2 improves over N1 in identifiable matched physics.

Then verify whether the improvement persists on new-only filters.

Do not assume it extends to mismatched physics.

### Gate 4 — Trust robustness

Evaluate whether any adaptive or constrained trust mechanism preserves the matched-physics advantage without substantial degradation under model mismatch.

Do not select a gate based only on aggregate performance across an arbitrary synthetic mixture of easy and difficult tasks.

Report conditional outcomes.

### Gate 5 — Real-survey readiness

A future real-data experiment is justified only if:

- Physical-score construction is interpretable.
- Matched-convergence comparisons support a useful effect.
- The effect is not confined to exact synthetic template matching.
- There is an adequate plan for independent spectroscopic supervision.
- Real catalogue preprocessing and filter-response limitations are understood.

Passing these gates does not prove CLAUDS transfer.

## 13. Resource discipline

Use compact numerical networks and existing forward kernels.

Avoid large hyperparameter sweeps.

A modest source-only temperature/TEF diagnostic is allowed.

A small gate may be trained only after fixed controls show a reason to introduce one.

Prefer paired seeds and independent SED identities over increasing model complexity.

No Qwen or other LLM training.

No Stage 2C.

Do not interrupt active GPU jobs.

## 14. Deliverables

Create:

`filter-aware-runs/1028/stage2b6-physical-score-trust/`

or an equivalent new non-overwriting directory.

Produce:

1. Stage 2A.6 handoff verification.
2. Physical-score and TEF audit.
3. Matched-convergence N1/N2 training records.
4. Fixed physical-score ablations.
5. Conditional trust-model implementation, if justified.
6. Matched and mismatched SED results.
7. Held-out filter and held-out family evaluations.
8. Paired bootstrap and training-seed uncertainty.
9. Source-only real-data plausibility diagnostics where feasible.
10. PDF-calibration and physical-adequacy plots.
11. Parameter budgets, training cost and inference runtime.
12. Reproducible code, configuration files, seeds and input hashes.
13. A concise scientific report.

End with explicit answers:

1. Does N2 outperform a properly converged N1 under matched source supervision?
2. Is the decoder benefit still present on strictly new filter responses?
3. How much of the earlier N2 behaviour came from physics-score tempering and TEF?
4. Can a simple physically motivated trust rule preserve useful likelihood information while limiting harm from SED mismatch?
5. Does marginalised or constrained physics from Stage 2A.6 improve on unrestricted profile physics?
6. Does the result justify a larger real-survey experiment?
7. Is there any demonstrated reason to introduce pretrained Qwen in Stage 2C?

**Execute all justified lightweight experiments. Do not stop after writing a proposal. Do not use target labels for model selection. Do not claim general physical transfer from a matched-template synthetic success alone.**