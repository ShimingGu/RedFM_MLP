# Paper II — Anecdote: Frozen-Backbone Head-Only Representation Transfer

## Scientific objective

Investigate whether photometric-redshift posttraining creates transferable representations in Qwen3.5-4B models, even when their original photo-z prediction heads fail under cross-survey distribution shift.

The central scientific question is:

**Does photo-z posttraining improve the sample efficiency and generalisation of a newly trained redshift head on another photometric survey?**

This experiment must distinguish:

1. Transfer of pretrained and posttrained backbone representations.
2. Transfer of an existing redshift prediction head.
3. Learning a new target-domain redshift mapping from limited supervision.
4. Memorisation or prior exposure to downstream objects.
5. Effects of survey-specific input serialization.

The experiment is deliberately a **frozen-backbone, head-only** study.

No Qwen backbone or LoRA/DoRA adapter parameters may be updated.

It belongs to the Paper II CLAUDS-transfer branch. Do not merge its scientific objectives or execution dependencies with the separate Paper I HSC–Legacy/high-z research branch.

## 1. Experimental hypothesis

Consider three backbone families:

**B0 — Original foundation model**

The original Qwen3.5-4B-Base checkpoint before any photo-z posttraining.

**BS — Spectroscopic-redshift-posttrained model**

An existing Qwen3.5-4B checkpoint with photo-z posttraining on a verified spectroscopic-redshift source catalogue.

Potential historical checkpoints include HSC-trained, Legacy-trained, QLoRA and DoRA models.

**BC — CLAUDS-posttrained model**

An existing Qwen3.5-4B checkpoint posttrained on CLAUDS photometric data using its historical ZPHOT supervision.

First verify the exact checkpoint provenance.

Do not assume all reported checkpoints use the same base model, source catalogue, adapter architecture or redshift targets.

Prioritise one clearly identifiable, comparable checkpoint for each family.

Additional historical spec-z checkpoints may be included as separately labelled arms if they are genuinely comparable and computationally inexpensive.

### Principal downstream tasks

**Task T-GMS:**

Train a fresh head on GalaxiesML-Spec HSC photometry with DESI spectroscopic-redshift labels.

**Task T-CLAUDS:**

Train a fresh head on CLAUDS photometry using limited target-domain supervision.

Separate two possible CLAUDS supervision types:

- T-CLAUDS-ZPHOT: historical Phosphoros ZPHOT labels.
- T-CLAUDS-SPEC: independently matched spectroscopic redshifts.

Do not combine these label types into a single training cohort without a separately defined experiment.

### Main comparison

| Frozen backbone | T-GMS head-only | T-CLAUDS head-only |
|---|---|---|
| B0 Original Qwen | Baseline | Baseline |
| BS Spec-z posttrained | Potential source-domain control | Cross-survey transfer |
| BC CLAUDS posttrained | Cross-survey transfer | Potential source-domain control |

The primary scientific comparisons are the off-diagonal transfer directions.

If BS was originally posttrained on GalaxiesML-Spec, BS→GMS is an in-domain control rather than independent cross-survey transfer.

If BC was originally posttrained on CLAUDS, BC→CLAUDS is an in-domain representation-reuse control, even though a new head is trained.

Do not describe these diagonal experiments as novel cross-survey evidence.

## 2. Read and verify the existing experiment history

Inspect previous reports and checkpoint manifests from:

- Historical Qwen3.5-4B photo-z QLoRA runs.
- Historical Qwen DoRA runs, where relevant.
- CLAUDS photo-z posttraining.
- GalaxiesML-Spec photo-z training.
- Stage 0 and Stage 0.5 representation/serialization audits.
- Stage 1 frozen-pretrained-versus-random experiments.
- Stage 3A and Stage 3B catalogue identity and spectroscopic-validation manifests.

Locate the actual model files.

For each candidate checkpoint, verify:

- Base-model name and revision.
- Base-model hash.
- Adapter type and rank.
- Adapter and head checkpoint hashes.
- Original source dataset.
- Original training labels.
- Original train/validation/test membership.
- Original photometric representation.
- Input serialization.
- Model hidden dimension.
- Feature pooling method.
- Original redshift grid.
- Whether the backbone itself was modified.
- Whether only adapters and redshift heads were trained.

Do not infer these properties from filenames.

If checkpoint compatibility cannot be proven, exclude the incompatible checkpoint rather than silently comparing different model families.

Do not retrain a missing checkpoint.

## 3. Critical source-exposure and identity-leakage audit

Before extracting downstream features, determine which galaxies each historical posttrained model previously encountered.

This is mandatory.

### Original training exposure

For BS and BC, recover their historical training identities and redshift-label provenance.

A model trained on CLAUDS ZPHOT may have already seen galaxies later appearing in:

- GalaxiesML-Spec.
- The 1,976-object CLAUDS spectroscopic cohort.
- Other downstream photometric evaluation sets.

Likewise, a historical spectroscopic source model may have encountered downstream targets through shared galaxy identities.

Audit:

- Exact catalogue identifiers.
- Coordinates and positional crossmatches.
- Duplicate photometric observations.
- Multiple views of the same galaxy.
- Historical source training membership.
- Historical source validation membership.
- Exposure to ZPHOT pseudo-labels.
- Exposure to genuine spectroscopic labels.

Distinguish direct training exposure from merely appearing in an archived validation file.

### Required cohort categories

Create at least:

**Clean transfer cohort**

Downstream objects not exposed to any compared posttrained model through its historical supervised training.

**Historical-exposure diagnostic cohort**

Objects with known prior exposure, clearly marked as non-independent for representation-transfer claims.

**Uncertain-exposure cohort**

Objects with unresolved training provenance or identity ambiguity.

Exclude uncertain-exposure objects from the primary interpretation.

For the principal B0/BS/BC comparison, use exactly the same eligible downstream galaxies for all three backbones.

Avoid constructing different test populations for different checkpoint families.

If a sufficiently large clean cohort cannot be formed, report that limitation and conduct an explicitly exploratory experiment instead.

The existing Stage 3A T2 spectroscopic identities were reserved for the earlier Stage 3 experiment, but their outcomes have since been inspected.

They are not a newly untouched confirmation set for this anecdotal study.

A new support/query split within T2 can be used for exploratory few-shot evaluation, but a future confirmatory result requires a genuinely independent cohort or a separately justified protocol.

## 4. Frozen feature-extraction contract

The central experimental operation is:

`h_i = FrozenBackbone(tokens_i)`

followed by:

`p(z | x_i) = FreshHead(h_i)`

All backbone and adapter parameters must remain frozen.

### Model configurations

**B0**

Original Qwen3.5-4B-Base with no photo-z adapter.

**BS**

The identical compatible base checkpoint plus its spectroscopic photo-z adapter.

**BC**

The identical compatible base checkpoint plus its CLAUDS photo-z adapter.

Do not use any historical learned redshift head in the main head-only comparison.

All new heads start from fresh, controlled initialization.

### Frozen-backbone integrity

Verify:

- Backbone parameters have `requires_grad=False`.
- Adapter parameters have `requires_grad=False`.
- No optimizer parameter group contains backbone or adapter parameters.
- Backbone and adapter remain in evaluation mode.
- No inference dropout or uncontrolled stochasticity is active.
- Hidden-state extraction uses the original compatible tokenizer.
- Numerical precision is documented.
- The correct final nonpadding token is selected.
- Attention masks and padding side are handled consistently.
- Any BF16/autocast settings necessary for faithful checkpoint execution are preserved.
- Model/adapter hashes remain unchanged after every experiment.

Where feasible, independently replay a small selection of archived native predictions before feature extraction to verify that each historical checkpoint is loaded correctly.

### Feature pooling

Use the historical last-token representation as the primary readout when compatible with the checkpoint.

Stage 0.5 found substantial sensitivity to terminal-token identity.

A separately predeclared masked-mean pooling control may be included.

Do not choose the pooling method by inspecting downstream test performance.

If pooling changes the hidden-state dimension or statistics, keep the fresh head comparison consistent.

### Optional negative control

If resources allow, include a compatible random-initialised adapter or another well-defined untrained adapter control.

This distinguishes learning-dependent representation changes from effects caused merely by inserting an adapter architecture.

Do not treat a randomly initialised 4B backbone as necessary for this initial study.

## 5. Target input serialization

Serialization is a major possible confound.

The previous Qwen experiments demonstrated strong sensitivity to:

- Numerical formatting.
- Band order.
- EBV placement.
- Terminal token.
- Flux amplitude.
- Missing-field representation.

Use one fixed, physically valid downstream input protocol for all compared backbones within a given task.

For T-GMS, all B0/BS/BC backbones must receive identical photometric input strings.

For T-CLAUDS, all B0/BS/BC backbones must likewise receive identical strings.

Do not permit each pretrained adapter to receive a different amount of information.

### Primary protocol

Use a documented canonical photometric serialization containing:

- Actual observed band identity.
- Flux.
- Uncertainty.
- Availability.
- Independently known metadata when applicable.

Preserve physically meaningful signed fluxes.

Do not insert fabricated measurements for unavailable bands.

Do not pretend EBV=0 when EBV is unavailable.

Keep a fixed terminal marker or format that is identical across compared backbones.

### Optional compatibility control

Because BS and BC may have been trained with different source-native serializations, perform one limited, predeclared formatting control where feasible.

For example, compare:

- Target-native canonical serialization.
- A shared alternate serialization compatible with historical feature extraction.

The same alternate strings must be evaluated across all backbones.

Do not use different source-native prompts for different models in the primary comparison.

Report whether an apparent transfer advantage depends strongly on input formatting.

Avoid optimizing serialization against target test labels.

## 6. Feature caching

Feature extraction is the computationally expensive stage; head training should be lightweight.

Extract each backbone's hidden states once per eligible object and serialization condition.

Cache features with metadata including:

- Object identifier.
- Dataset and partition.
- Backbone hash.
- Adapter hash.
- Tokenizer hash.
- Serialization version.
- Pooling type.
- Feature dimension.
- Numerical precision.
- Input checksum.
- Extraction code version.

Check deterministic repeatability.

Use the same object order and identity mapping for each backbone.

Do not mix cached representations from different checkpoint or serialization versions.

For the principal frozen-feature comparison, compute standardisation parameters using downstream head-training data only.

A shared source-defined feature-normalisation protocol is acceptable.

Do not fit feature normalisation using held-out query labels.

If unsupervised full-cohort statistics are used in an explicitly transductive experiment, label that experiment separately.

## 7. Fresh prediction-head architecture

All backbones must receive exactly the same newly initialised prediction-head architecture.

Use a small, computationally practical head, for example:

`hidden_dim → 32 → 300 redshift bins`

with a documented activation and regularisation scheme.

Use the same output bins, loss definition and training protocol for all backbones.

Primary redshift grid:

`0 <= z <= 4.5`, 300 bins.

Verify source/target redshift membership before training.

Do not clip labels outside the grid into the boundary bins.

### Head-only optimisation

Train only the new head.

Use:

- Same hidden dimension for compatible backbones.
- Same head architecture.
- Same head initialization seeds.
- Same downstream training identities.
- Same optimizer family.
- Same batch construction.
- Same training-label budget.
- Same permitted source-validation checkpoint-selection procedure.

Head-specific regularisation may be selected on development data using a small predeclared configuration set, applied symmetrically to all backbones.

Do not give historically posttrained backbones a larger or better-tuned head.

### Sparse-redshift supervision

With K=16–128 training objects and 300 redshift bins, many output bins receive no direct training label.

This was a significant issue in Stage 2B.5.

Include a fixed, clearly declared treatment of sparse labels.

For example, use a small Gaussian-neighbour soft-label target as a separate control.

Do not change the evaluation grid or probability convention silently.

Use hard-label CE as the main reference when feasible.

Any head-only smoothing or calibration must use training and development labels only.

Do not use target test performance to choose smoothing strength.

## 8. Few-shot supervision budgets

The main experiment is a learning curve across target supervision budgets.

Use:

`K = 16, 32, 64, 128, 256, 512`

Optionally include K=1024 when sufficient independent support data exist.

For each dataset:

- Create nested calibration/training subsets.
- Use identical subsets for every backbone.
- Repeat subset selection across multiple fixed random seeds.
- Repeat head initialization independently.
- Use a fixed downstream development set.
- Use one fixed independent query/test set for all K.

Do not draw fresh test objects for every K.

Report uncertainty from:

1. Calibration-subset selection.
2. Head initialization.
3. Galaxy-level query sampling.

Do not mistake repeated training heads on the same spectroscopic galaxy for independent test galaxies.

### K=0 distinction

A newly random-initialised supervised head at K=0 is not a meaningful photo-z estimator.

If reporting K=0, use the historical frozen original-head zero-shot models as a separately labelled reference.

Do not combine K=0 original-head predictions with K>0 fresh-head predictions into one continuous experimental curve without clearly identifying the change in prediction head.

The main head-only learning curve starts at K>0.

## 9. T-GMS evaluation

Use GalaxiesML-Spec with its verified HSC grizy photometry and DESI spectroscopic redshifts.

Use the actual historical source partitions and object-exposure audit.

The primary transfer question is whether BC's CLAUDS-posttrained features improve target head-only learning over B0.

If a BS model was originally trained on HSC/GalaxiesML-Spec, treat its GMS performance as an in-domain control.

If a distinct Legacy-trained BS checkpoint is available, its transfer to GMS may provide an additional genuinely cross-survey comparison, subject to identity exclusions.

For each K, evaluate:

- B0→GMS.
- BS→GMS.
- BC→GMS.

Use the same eligible GMS spectroscopic test galaxies.

Separate high-z strata only where there is adequate spectroscopic support.

Do not equate within-source sample efficiency with cross-survey representation transfer.

## 10. T-CLAUDS evaluation

Treat CLAUDS carefully because historical ZPHOT supervision and independent spectroscopy are different labels.

### Experiment C-ZPHOT

Train fresh heads using CLAUDS ZPHOT.

This tests how efficiently the frozen representations learn the CLAUDS photometric-reference mapping.

It does not by itself demonstrate true-redshift accuracy.

Where possible, restrict the main test objects to those not previously seen by BC during historical ZPHOT training.

If clean label-unexposed objects are unavailable, report that the result is in-domain representation reuse rather than independent transfer.

### Experiment C-SPEC

Train fresh heads using a limited number of high-quality CLAUDS spectroscopic matches.

Use the 1,976-object Stage 3A/3B T2 cohort only under a new, fixed, explicitly exploratory support/development/query split.

Preserve exact identity grouping and exclude historical-source training exposure.

A reasonable protocol may allocate enough objects to:

- A fixed query cohort.
- A separate development cohort.
- A maximum K=512 nested calibration pool.

Lock these partitions before running the new head-only learning curves.

Avoid using the query cohort for configuration, epoch, feature format, or head selection.

A fresh independent spectroscopic cohort will eventually be required for a new confirmatory claim.

For each K compare:

- B0→CLAUDS.
- BS→CLAUDS.
- BC→CLAUDS.

The main cross-survey comparison is BS versus B0.

BC→CLAUDS is primarily an in-domain adapter control.

### Spectroscopic selection

The current T2 spectroscopic population is substantially brighter than the local unlabeled CLAUDS population.

Report that limitation.

Stratify by:

- Apparent magnitude.
- S/N.
- Photometric completeness.
- Source-support diagnostics.
- Redshift.

Do not claim that results on the bright T2 subset generalise to the full faint CLAUDS population.

The small number of z>=2 T2 objects is not sufficient for robust high-z conclusions.

## 11. Mandatory non-foundation-model controls

A positive head-only result should demonstrate more than the ability to fit K target labels.

Include inexpensive controls using the same support and query galaxies.

### Direct photometric kNN

Perform numerical nearest-neighbour regression in an uncertainty-aware target photometric feature space.

Return a redshift PDF or explicitly label the result as point-only if no well-defined PDF is available.

### Compact numerical MLP

Train a conventional target-domain photo-z model from the same K labels, with identical query identities.

### Global target correction

Where a source-trained photo-z prediction is available, compare a simple global source-to-target redshift calibration.

This controls whether a more complicated feature-transfer method merely corrects a global bias.

### Fixed pretrained representation

B0 is the essential foundation-model control.

All additional numerical baselines must have their supervision and physical information access documented.

Do not compare a 512-label head-only model to a zero-label numerical model as if they had equal supervision.

## 12. Metrics and statistical interpretation

For every eligible evaluation condition report:

- NMAD.
- Catastrophic-outlier fraction.
- Median bias.
- Mean bias.
- Cross-entropy.
- CRPS.
- Nominal 68% coverage.
- PIT diagnostics.
- Predicted-redshift distribution.
- Narrow-interval concentration.
- Unsupported or failed predictions.

Use the historical catastrophic-outlier definition:

`abs(z_pred-z_true)/(1+z_true) > 0.15`

Specify the point estimator derived from the PDF.

Report the same metrics for all three frozen backbones on identical test objects.

### Primary representation-transfer statistic

For each target and K, report:

`Delta_outlier(BS or BC, B0)`

and:

`Delta_CE(BS or BC, B0)`

Use paired test-galaxy resampling.

Calculate intervals conditional on calibration set and head initialization, then separately report variation across subset seeds and head initialization seeds.

A statistically convincing result should not depend on one fortunate head initialization.

### Learning-curve summary

Plot each backbone's performance versus K.

Quantify:

- Small-K transfer gain.
- Largest observed gain.
- Stability across K.
- Stability across dataset splits.
- Whether models converge to similar performance at large K.
- Whether any posttrained backbone exhibits negative transfer.
- Relative sample efficiency at predeclared error thresholds, where statistically supported.

Do not choose a favourable K after inspecting all results and present it as the sole primary endpoint.

Predeclare K=64 and K=256 as primary small/medium-budget comparisons if the cohort sizes support them.

Other K values form the learning curve and secondary analysis.

### Scientific interpretation

If BS or BC outperforms B0 using fewer labels, this supports a transferable downstream representation under the tested readout and input protocol.

It does not by itself prove that the advantage is specifically astrophysical physical knowledge.

If performance degrades, this supports a negative-transfer finding for the tested representation, but does not establish that photo-z posttraining erased pretrained knowledge.

Input serialization, source exposure, feature pooling and optimisation remain possible confounders.

## 13. Control for historical-head effects

This experiment is designed to separate backbone representation from prediction-head transfer.

For each previously posttrained checkpoint where the original head is available, preserve an optional comparison:

**Original frozen backbone + original frozen head**

versus:

**Same frozen backbone + fresh target-trained head**

Report these under separate labels.

Do not allow original-head parameters to be used to initialise the new head in the primary experiment.

The distinction is:

`representation transfer != zero-shot prediction transfer`

A posttrained backbone may contain useful target-domain information even when the source-trained prediction head fails.

Likewise, a successful original source head does not demonstrate a universally useful representation.

## 14. Execution gates

### Gate 0 — Checkpoint provenance

Compatible base models and adapter checkpoints are verified.

If a checkpoint is missing, do not substitute a different architecture without marking the new experiment.

### Gate 1 — Historical exposure audit

Primary comparisons use source-exposure-compatible downstream cohorts.

If this is impossible, execute an explicitly exploratory analysis with clear contamination labels.

### Gate 2 — Feature extraction

Frozen hidden-state extraction is numerically reproducible and produces finite aligned features.

No backbone or adapter parameter changes.

### Gate 3 — Head learnability

Fresh heads demonstrate genuine training and validation learning on at least a moderate K.

If the models select update zero across all heads, diagnose sparse-label supervision and optimiser behaviour rather than interpreting the result as evidence about backbone quality.

### Gate 4 — Head-only transfer comparison

Compare B0/BS/BC on identical eligible query galaxies with multiple head and subset seeds.

No test-label-based checkpoint or configuration selection.

### Gate 5 — Generalisation interpretation

Report separately:

- Clean cross-survey representation transfer.
- Historical-exposure or in-domain adaptation.
- ZPHOT-reference agreement.
- Genuine spectroscopic target performance.
- Unsupported high-z or faint-regime extrapolation.

Do not upgrade exploratory results into independent confirmation.

## 15. Resource discipline

No backbone fine-tuning.

No QLoRA or DoRA updates.

No large-model training sweep.

Use cached frozen features, then run lightweight head experiments.

Start with a deterministic small pilot on K=64 and K=256 to verify head learnability and runtime.

The full learning-curve protocol and eligible cohorts must be locked before opening query-label evaluations.

Expand to the full predefined budget curve only after the correctness and learnability gates pass.

Prefer multiple small-head seeds over excessive optimisation of any one head.

Do not interrupt active GPU jobs or overwrite previous checkpoints.

## 16. Deliverables

Create a new experiment directory such as:

`filter-aware-runs/1029/stage4-frozen-backbone-transfer/`

Include:

1. Complete historical checkpoint provenance.
2. Historical supervised-exposure audit.
3. Clean and exploratory downstream cohort manifests.
4. Shared input-serialization definitions.
5. Frozen hidden-feature extraction and verification.
6. Cached feature manifests and hashes.
7. Fresh-head architecture and training code.
8. Nested K-budget experiment manifests.
9. T-GMS learning curves.
10. T-CLAUDS-ZPHOT learning curves, where eligible.
11. T-CLAUDS-SPEC exploratory learning curves.
12. Numerical kNN/MLP/global-calibration controls.
13. Historical-head versus fresh-head comparisons.
14. Paired bootstrap and training-seed variability.
15. Runtime, memory and caching costs.
16. Source/target conditional metrics and PDF calibration.
17. Full reproduction commands.
18. A concise scientific report.

## 17. Required final assessment

Answer explicitly:

1. Does BS transfer better than the original B0 backbone to CLAUDS under the same few-shot head-only supervision?

2. Does BC transfer better than B0 to GalaxiesML-Spec?

3. Does photo-z posttraining improve the learning curve at small target-label budgets?

4. Are any improvements stable across target subset selection, head initialization and feature pooling?

5. How much of the apparent improvement is explained by prior exposure to downstream objects?

6. Are the findings compatible with improved transferable representations rather than simply retaining a successful original head?

7. Does any advantage remain when compared with simple numerical kNN and target-only MLP controls?

8. Is any advantage sensitive to source-native versus shared target serialization?

9. Does transfer differ between CLAUDS ZPHOT learning and independent spectroscopic-redshift learning?

10. Does the observed benefit support future few-shot prompting or calibration experiments without additional backbone posttraining?

11. Which models and clean downstream cohorts are suitable for the next Paper II experiment?

12. What additional evidence would be necessary to claim that the transferred information is genuinely astrophysical pretraining knowledge?

## 18. Execution instructions

Execute the experiment rather than stopping at a design report.

First complete checkpoint and object-exposure audits.

Then implement the common frozen-feature extractor and verify no backbone/adaptor parameter changes.

Execute a small K=64/256 pilot using source/development-only checkpoint selection.

If correctness and head-learnability gates pass, run the complete predefined learning curves and baseline controls.

Use all three backbone families when valid checkpoints exist.

Do not silently omit a transfer direction because it performs poorly.

Do not invent clean cohorts where training histories are unavailable.

Do not use test labels to optimise heads, serialization, pooling, regularisation, training steps or feature extraction.

Maintain historical experiment integrity.

Do not start Stage 2C, Qwen fine-tuning, new large-model pretraining or unrelated physics-kernel experiments.

**The primary outcome is evidence for or against transferable photo-z representations after posttraining—not merely the best achievable target-domain NMAD.**

Report positive, negative and inconclusive results with equal care.