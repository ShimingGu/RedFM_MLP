# Stage 1 — Does Qwen Actually Use Pretrained Astrophysical Knowledge?

## Scientific question

We are investigating whether general foundation models contribute astrophysical knowledge to photometric-redshift inference, rather than merely functioning as expensive nonlinear regressors.

Our existing Qwen3.5-4B-Base + QLoRA models are strong in-domain photo-z estimators but fail on CLAUDS cross-survey transfer.

Adding filter names, pivot wavelengths and bandwidths did not resolve the CLAUDS failure.

These observations do not establish whether the pretrained model possesses or uses astrophysical knowledge.

**Design and execute controlled experiments that distinguish physical reasoning, generic function approximation and benefits of pretrained representations.**

This is primarily a diagnostic benchmark, not another full-scale photo-z training project.

## 1. Inspect available resources

Inspect the existing:

- Qwen3.5-4B-Base model.
- Frozen-Qwen inference and embedding extraction pipeline.
- Existing photo-z PDF head.
- Filter response registry and original transmission profiles.
- Synthetic photometry or SED-template utilities, if already available.
- Existing MLP and TabICL evaluation infrastructure.

Reuse suitable components.

Do not alter archived checkpoints.

Do not use CLAUDS target redshift labels to design, tune or select these experiments.

## 2. Construct a physically controlled SED benchmark

Generate synthetic galaxy spectra with explicitly controlled physical properties.

Include at least:

- A 4000 Å/Balmer-break family.
- Lyman-break/IGM-absorption effects where physically appropriate.
- Different continuum slopes.
- Dust attenuation.
- Multiple spectral-feature strengths.
- Smooth-spectrum controls with weak or absent breaks.

Use realistic stellar-population or galaxy SED templates where accessible, alongside simple analytic templates whose behaviour is completely understood.

For an intrinsic SED and redshift, generate synthetic observations by redshifting the spectrum and integrating it through the actual filter response curves.

Do not approximate the measurement by simply sampling the SED at each filter's pivot wavelength.

Handle AB flux-density conventions, throughput weighting, cosmological flux scaling and filter normalisation consistently.

Treat the overall luminosity/normalisation as a nuisance variable when redshift information comes from spectral shape.

Generate HSC, Legacy and CLAUDS-like photometric observations from the same underlying SEDs.

Use physically consistent noise models and include controlled missing-band scenarios.

Save all simulation parameters, spectra, filters, truth labels and random seeds.

## 3. Design three independent physics probes

### Probe A — Explicit spectral reasoning without photo-z fine-tuning

Test whether a language-model interface can use known spectral features.

Examples:

- Given a rest-frame break at 4000 Å and its observed wavelength, infer redshift.
- Distinguish a Balmer/4000 Å break from a Lyman break where the observations are ambiguous.
- Reason about how a known spectrum should appear in different filter systems.
- Identify whether the supplied redshift, spectral feature and wavelength information are physically consistent.

Include simple arithmetic and rule-based baselines.

Test no-example and few-example conditions separately.

Important: Qwen3.5-4B-Base is a base checkpoint, not necessarily an instruction-following model. Do not interpret poor free-form instruction following as evidence that the pretrained representation lacks physical information.

If a compatible Instruct model is available, test it as a separate secondary condition. Do not conflate instruction tuning with astrophysical pretraining.

### Probe B — Frozen representation test

Keep the pretrained Qwen backbone frozen.

Extract representations for synthetic photometry plus correct filter information.

Train an identical small probe/head on a limited labelled synthetic set and evaluate on independent synthetic objects.

Use a strict matched protocol for:

- Pretrained frozen Qwen representations.
- Random-initialised frozen Qwen architecture representations, if computationally feasible.
- A parameter-budget-controlled MLP on numerical inputs.
- Existing suitable tabular baselines.
- An explicit physical/template-fitting baseline.

A randomly initialised *frozen* Qwen is a representation control, not a complete comparison of pretrained versus fully trained architectures.

Keep the training sample count, probe architecture, evaluation support and optimiser budget matched.

Do not allow the pretrained backbone to adapt in this primary comparison.

Repeat a small number of seeds for configurations with promising differences.

The objective is to determine whether pretrained representations provide an advantage beyond arbitrary nonlinear random features and learned tabular regression.

### Probe C — Physically controlled cross-filter transfer

This is the most important probe.

Train or calibrate only on synthetic observations using selected source filter systems.

Then evaluate on new observations of independent SEDs through a held-out filter system.

Use physically paired counterfactuals:

- Same intrinsic SED.
- Same redshift.
- Same intrinsic nuisance parameters.
- Different observational filter response.
- Consistently regenerated fluxes and uncertainties.

The correct redshift should remain unchanged when only the observational system changes.

Do not give the model the answer, intrinsic SED identity or simulation seed.

Also test controlled redshift changes for the same intrinsic SED, where the expected spectral-feature movement is known.

Measure how predictions respond to:

1. Physically correct filter information.
2. Shuffled or incorrect filter metadata with fluxes held fixed.
3. Physically self-consistent changes of filter response and photometry.
4. Missing or noisy filters.

Distinguish a reaction to incorrect tokens from evidence of physically consistent reasoning.

A model becoming worse when metadata is shuffled does not, by itself, prove that it understands the physics.

## 4. Prevent trivial shortcuts

The benchmark must not be solvable only through:

- Survey name.
- Fixed filter position.
- Total brightness.
- Input string length.
- Distinct noise levels assigned to different redshifts.
- Repeated SEDs across train and evaluation splits.
- Repeated nuisance parameters correlated with redshift.

Split by underlying SED identity before generating multiple redshifts, noise realisations or filter observations.

Include a stronger generalisation test holding out SED families or spectral-feature combinations.

In the main cross-filter benchmark, the target filter system must not appear during supervised training.

Include a shuffled-redshift-label control to check for unexpected leakage.

## 5. Evaluate physical behaviour, not just NMAD

Report:

- Redshift NMAD and catastrophic-outlier fraction.
- Bias as a function of redshift.
- CE and CRPS for probabilistic models where available.
- Conditional results by SED family, redshift and S/N.
- Performance on previously unseen filters.
- Performance on previously unseen SED families.
- Paired prediction differences when the same SED is re-observed through another filter system.
- Performance against an analytic or template-fitting physics baseline.

For the explicit reasoning probe, additionally report correct/incorrect answers and physical consistency.

Do not force scalar-output models and PDF-output models into an artificially identical metric without documenting the distinction.

## 6. Test the pretrained-knowledge hypothesis

Separate three claims:

**Claim 1: Physical behaviour is present.**

The model's predictions respond consistently to physically controlled changes in redshift and filter response.

**Claim 2: Pretraining provides a benefit.**

Frozen pretrained representations outperform matched random-feature and supervised tabular controls on controlled held-out physics tasks.

**Claim 3: Pretrained astrophysical knowledge is specifically responsible.**

This stronger claim requires evidence beyond generic pretraining benefits, such as a carefully designed control in which the required astrophysical relation is unavailable from the supervised examples but can plausibly be inferred from pretrained knowledge.

Be explicit about which claims the experiments can and cannot support.

Generic numerical reasoning, general pretraining and domain-specific astrophysical knowledge are not automatically equivalent.

## 7. Resource discipline

Start with small inference-only probes.

Then run frozen-representation probes on limited synthetic datasets.

Do not launch full QLoRA or DoRA training unless the cheaper experiments produce interpretable evidence.

Prefer a well-controlled small experiment over a large benchmark matrix.

If random-initialised full-model experiments are too expensive, implement the feasible frozen-architecture control and document its limitations.

Do not infer lack of knowledge merely because a particular prompt fails.

## 8. Deliverables

Create an independent Stage-1 directory containing:

- Synthetic SED generator and forward-model tests.
- Filter-throughput integration tests.
- Fixed simulation and split manifests.
- Explicit reasoning probe results.
- Frozen pretrained-versus-control comparisons.
- Cross-filter invariance and redshift-response plots.
- Per-SED-family and per-redshift metrics.
- Source code, exact configurations and reproduction commands.
- Scientific interpretation separating evidence for physics, pretraining and astrophysical knowledge.

End the report by answering:

1. Does pretrained Qwen exhibit usable physical information under any tested interface?
2. Is there evidence this information comes from pretraining?
3. Can the model transfer spectral-redshift relations to unseen filter systems?
4. Does this justify introducing a physics-guided Qwen architecture in Stage 2?

**Execute the feasible low-cost probes and return empirical results, not only a proposed benchmark.**

Do not claim success on real CLAUDS from synthetic results alone.