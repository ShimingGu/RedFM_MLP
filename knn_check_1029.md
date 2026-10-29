# Paper II — Stage 4.5: Why Does a Spec-z-Posttrained Qwen Lose to Photometric kNN?

## Scientific objective

Investigate an unexpected discrepancy in the completed Stage 4 frozen-backbone representation-transfer experiment.

On GalaxiesML-Spec (GMS), with K=256 downstream head-training galaxies:

- B0 (original Qwen3.5-4B-Base), hard-label head: 28.68% catastrophic outliers.
- BS (spec-z-posttrained Qwen), hard-label head: approximately 21.13%.
- BC (CLAUDS-ZPHOT-posttrained Qwen), hard-label head: 22.67%.
- Numerical MLP: 17.06%.
- Direct photometric kNN: 2.54%.

BS improves substantially over B0, but direct photometric kNN is dramatically better.

The purpose of this experiment is to determine whether this discrepancy primarily reflects:

1. Discarding BS's previously trained redshift prediction head.
2. Failure of a fresh 300-bin head under limited downstream supervision.
3. Poor local redshift geometry in the frozen hidden-state representation.
4. An unusually easy or contaminated kNN benchmark.

These explanations are not mutually exclusive.

Do not assume in advance that the foundation-model representation is good, that kNN is leaking information, or that the readout is solely responsible.

This experiment belongs to **Paper II**, specifically the CLAUDS-transfer and representation-generalisation branch. Do not expand into the separate Paper I project.

## 1. Terminology and existing assets

Use the following definitions throughout:

- **B0:** Original Qwen3.5-4B-Base without photo-z posttraining.
- **BS:** Qwen3.5-4B-Base with a previously trained spectroscopic-redshift photo-z adapter.
- **BC:** Qwen3.5-4B-Base with a previously trained CLAUDS ZPHOT photo-z adapter.
- **GMS:** GalaxiesML-Spec, using HSC grizy photometry and independently verified DESI spectroscopic-redshift labels.

The primary experiment concerns **BS on GMS**.

BC and B0 remain useful controls but should not distract from the primary discrepancy.

Read the completed Stage 4 report, checkpoint provenance, historical-exposure audit, feature manifests, support/development/query splits, head-learning curves, and numerical-control outputs.

Reuse the identical Stage 4 clean GMS objects:

- 1,024 support-pool objects.
- 256 fixed development objects.
- 1,024 fixed query objects.

The principal comparison uses K=256 training labels drawn from the support pool.

Retain the same three support-subset seeds and two fresh-head initialization seeds, giving six paired configurations.

**Important:** K=256 means 256 gradient-training labels plus 256 labelled development objects. Do not describe it as 256 total labels.

The GMS query cohort was previously evaluated. This Stage 4.5 investigation is therefore a retrospective, exploratory diagnostic, not a new independent confirmation experiment.

Do not change the cohort or quietly redefine eligibility to improve results.

## 2. Global execution constraints

- No backbone fine-tuning.
- No LoRA/DoRA updates.
- Do not modify any historical checkpoint.
- Do not retrain BS's original redshift head.
- Do not introduce target query labels into training, calibration, feature normalization, metric design, or model selection.
- Reuse existing frozen-feature caches when their hashes and configurations are valid.
- Preserve the previously fixed support/development/query splits.
- Keep all compared models on the identical query galaxies.
- Do not merge support and development labels into a falsely claimed K-only experiment.
- Do not overwrite Stage 4 results.
- Do not introduce unrelated model architectures or large hyperparameter sweeps.

Perform all four checks below.

Source/development-only decisions may be made before evaluating diagnostic query results. Any configuration chosen after inspecting Stage 4 query performance must be explicitly described as retrospective.

---

# Check A — Original BS Backbone + Original BS Head + Native Serialization

## Scientific question

Does BS already perform well on the identical GMS query galaxies when its original trained redshift head and original input protocol are retained?

Stage 4 deliberately discarded every historical redshift head and trained a new one.

Consequently, Stage 4 primarily measured **new-head readability of frozen representations**, not the performance of the original trained BS photo-z system.

We need to separate these quantities.

## Required configurations

### A1 — BS original native pipeline

Load:

- The verified original BS base checkpoint.
- Its original trained adapter.
- Its original trained redshift head.
- Its original tokenizer.
- Its original GMS/HSC-compatible photometric serialization.

Use the checkpoint's native output grid and probability conventions.

Do not reinitialize the head.

Do not calibrate it using the GMS query labels.

Run inference on the same 1,024 clean GMS query galaxies used in Stage 4.

### A2 — BS original head with Stage 4 canonical serialization

Apply the same original BS adapter and original redshift head, but use the canonical Stage 4 photometric input serialization.

This isolates an important component of the serialization shift.

Only run A2 if the original head can validly consume the resulting hidden states without modification.

### A3 — Archived historical replay

Where possible, reproduce a small set of previously archived BS predictions under the historical native interface before running A1.

Verify exact or tolerance-consistent predictions and logit/PDF normalization.

If the original BS head cannot be recovered faithfully, report the blocker rather than silently constructing a replacement.

## Measurements

Report for A1 and A2:

- Catastrophic-outlier fraction.
- NMAD.
- Bias.
- CE.
- CRPS.
- Nominal 68% coverage.
- Predicted redshift distribution.
- Prediction concentration.
- Failure rate.
- Actual native redshift-bin range.

Also report how these figures compare with the Stage 4 fresh-head BS configuration on the identical query cohort.

The original head uses no new K labels, while the fresh head uses K plus development labels. This difference must be explicit.

Do not compare historical validation metrics directly with Stage 4 query metrics as if they were measured on the same objects.

### Interpretation

If native BS performs strongly but the new head performs poorly, that is consistent with a readout or serialization bottleneck.

It does not automatically establish that all information in the native prediction head is linearly available in the hidden states.

If A1 performs well but A2 collapses, investigate serialization sensitivity.

If both A1 and A2 perform poorly, examine whether the clean GMS query population differs from the original BS training and validation populations.

---

# Check B — Fresh BS Head with Gaussian Soft Labels

## Scientific question

Does the severe performance gap arise because 256 labels poorly supervise a fresh 300-class redshift classifier?

Previous Stage 4 results showed:

- K=64 hard-label readouts largely failed to learn.
- Several CLAUDS hard-label readouts selected initialization.
- For BC→GMS at K=256, Gaussian sigma=3-bin smoothing substantially reduced outliers relative to hard-label CE.

We need the corresponding **BS-specific** comparison, not an inference from BC.

## Required configurations

Use the identical cached BS features and fresh-head architecture from Stage 4.

Evaluate:

**B1 — Hard-label CE**

Reproduce the original K=256 head-only result.

**B2 — Gaussian soft-label CE**

Use fixed Gaussian smoothing with sigma=3 redshift bins.

The training target for each spectroscopic label is a normalized discretized Gaussian centered on its true redshift.

Handle grid boundaries consistently, renormalizing the soft target on the valid grid.

Use source/development-only checkpoint selection.

**B3 — Optional continuous regression readout**

If B1/B2 are fully reproduced and the implementation is straightforward, add one small scalar-redshift regression head as an explicitly secondary diagnostic.

It must use the same cached features, the same K labels, the same development labels, and the same query galaxies.

Predefine a loss such as Huber loss on a suitable redshift coordinate.

Do not introduce a large density-model architecture search.

A scalar prediction is point-only unless a separately specified probabilistic model is fitted. Do not report invented CE/CRPS for deterministic scalar output.

## Controls

Run B0 and BC under B1/B2 using exactly the same training configurations where cached features are available.

This separates:

- General improvement from soft-label learning.
- BS-specific representation transfer.
- Whether smoothing changes the BS–B0 difference.

Repeat the registered six support/head configurations.

Report:

- Training CE.
- Development CE.
- Best update.
- Whether the selected checkpoint differs from initialization.
- Query outliers/NMAD.
- Query CE/CRPS/coverage.
- Variation across six fits.

Do not choose the smoothing width using query results.

### Interpretation

The decisive question is whether BS's large original error is substantially removed by a better-supervised readout.

If soft labels greatly improve both BS and B0, but the BS–B0 gap remains small, the original failure is primarily a readout-design issue, not necessarily a representation-transfer success.

If BS improves much more than B0, that supports an interaction between BS representation and readout.

If all heads still struggle, we need to inspect the hidden-state geometry directly.

---

# Check C — kNN in Frozen BS Hidden-State Space

## Scientific question

Does the BS hidden-state representation preserve a local geometry that is useful for redshift estimation?

Compare:

`kNN(raw photometry)`

with:

`kNN(frozen BS hidden states)`.

Also include B0 and BC hidden states as controls.

This experiment tests **local neighbourhood geometry under specified distance metrics**, not the full recoverable information content of the representation.

A representation may be useful for a learned nonlinear decoder even if Euclidean kNN performs poorly.

## Required configurations

### C1 — Original photometric kNN

Reproduce the exact Stage 4 direct photometric kNN baseline.

Verify its feature construction and the reported 2.54% outlier fraction at K=256.

### C2 — BS hidden-state kNN

Use the frozen BS hidden states corresponding to the same GMS observations.

Fit kNN using only the K=256 support labels.

Evaluate on the identical 1,024 query objects.

### C3 — B0 hidden-state kNN

Repeat C2 using the original pretrained Qwen hidden states.

### C4 — BC hidden-state kNN

Repeat C2 using the CLAUDS-posttrained Qwen hidden states.

## Distances and normalization

Use a compact predeclared distance protocol.

At minimum examine:

- Euclidean distance on training-normalized hidden features.
- Cosine distance on appropriately normalized hidden features.

For high-dimensional representations, avoid unrestricted covariance inversion or fitting a high-capacity metric with only 256 labels.

If PCA or another unsupervised projection is used, fit it on the permitted support features only, or explicitly label a separate transductive protocol.

Do not use query redshifts when fitting the transformation.

If hyperparameters such as neighbour count, temperature or KDE bandwidth require selection, use a small fixed grid and development labels only.

Keep equivalent tuning access and label-budget accounting across the photometric and hidden-space kNN models.

## Neighbourhood diagnostics

For every query object, save:

- Nearest-neighbour identities.
- Distances.
- Neighbour spectroscopic redshifts.
- Query spectroscopic redshift, used only after prediction is frozen.
- Neighbourhood redshift dispersion.
- Effective neighbour count.
- Local prediction bias.
- Catastrophic-outlier status.

Report the association between representation distance and redshift distance.

Use meaningful summaries, such as redshift scatter among nearby neighbours and conditional outlier fraction by neighbour distance.

Avoid claiming that nearest neighbours in hidden space are physically similar solely because their Euclidean distance is small.

## Key comparisons

Evaluate:

1. Photometric kNN versus BS-hidden kNN.
2. BS-hidden kNN versus B0-hidden kNN.
3. BC-hidden kNN versus B0-hidden kNN.
4. BS-hidden kNN versus BS fresh-head predictions.

If possible, examine whether native BS redshift residuals correlate with poor hidden-space neighbourhood quality.

### Interpretation

If BS-hidden kNN is much better than the newly trained BS head, this supports a fresh-head optimization/readout bottleneck.

If raw-photometry kNN is much better than BS-hidden kNN, the chosen hidden-state distance metric is less redshift-informative locally than the raw photometric metric.

This does not prove that the BS backbone discarded all useful photometric information.

If BS-hidden kNN consistently beats B0-hidden kNN, that independently supports an effect of photo-z posttraining on neighbourhood geometry.

---

# Check D — Audit the Extremely Strong Photometric kNN Baseline

## Scientific question

Why does direct photometric kNN with K=256 achieve only approximately 2.54% catastrophic outliers on the clean GMS query cohort?

This result may be valid.

Do not assume leakage.

However, it is sufficiently strong relative to the neural controls that its difficulty and integrity deserve explicit verification.

## D1 — Exact provenance and leakage audit

Trace every feature used by Stage 4 photometric kNN.

Verify that no feature contains:

- True spectroscopic redshift.
- ZPHOT used as a target proxy.
- Predicted redshift from a supervised model.
- Redshift-dependent template-fit outputs.
- Redshift labels encoded through preprocessing.
- Query-label-derived normalization.
- Test-selected neighbour weights or hyperparameters.

Confirm that kNN support and query objects have:

- No shared catalogue identity.
- No repeated observation representing the same physical galaxy.
- No unresolved close spatial match.
- No accidental historical partition collision.
- No shared label-bearing preprocessing.
- No use of development/query redshifts in feature construction.

Check actual nearest-neighbour angular separations and photometric differences.

Do not automatically call physically close but distinct galaxies leakage.

Report ambiguous cases and their influence on performance.

If a bug or leakage is found, preserve the original output and publish the corrected result separately.

## D2 — Redshift-distribution difficulty

Measure the true redshift distribution of the frozen GMS query cohort.

Report:

- Number of objects.
- Min/median/max redshift.
- Redshift quantiles.
- Redshift histogram.
- Catastrophic-outlier fraction by redshift.
- Outliers versus source-training support.

Construct the following trivial baselines using only K support labels:

**Constant median**

`z_hat = median(z_spec of K support galaxies)`

**Constant mean**

`z_hat = mean(z_spec of K support galaxies)`

**Global K-label redshift distribution**

Build a simple, normalized redshift PDF from the K support labels without using query photometry.

Compare their point-outlier fractions with kNN.

This determines whether the apparently excellent kNN metric reflects a highly concentrated redshift distribution or genuinely informative photometric neighbourhoods.

## D3 — Feature ablation

Using the same support/query identities and fixed kNN evaluation code, compare small predeclared sets:

1. Colours only.
2. Absolute magnitude/flux only.
3. Colours plus magnitude.
4. Colours plus magnitude and measurement uncertainties.

Preserve exact treatment of missing bands and signed flux.

Any feature scaling or tuning must use only support/development data.

This is a diagnostic, not a new optimized kNN benchmark.

Determine whether the low outlier fraction relies mainly on:

- Spectral colours.
- Apparent-brightness/redshift correlation.
- Measurement-quality or selection information.
- A particular numerical feature engineering choice.

## D4 — Neighbour-redshift validity

For the final original photometric kNN model, report:

- Median nearest-neighbour photometric distance.
- Distribution of nearest-neighbour redshift differences.
- Fraction of nearest neighbours within several physical redshift tolerances.
- Outlier fraction as a function of neighbour distance.
- Outlier fraction as a function of apparent magnitude.
- Performance at different redshift intervals.
- Stability across the same three K=256 support subsets.

If a simple colour-distance kNN genuinely achieves approximately 2.54% outliers, explain what the query population and feature-space structure make possible.

Do not infer that the same performance will transfer to faint CLAUDS galaxies.

GMS and faint CLAUDS have different observational distributions.

---

# 3. Integrated comparison

After completing A–D, produce one comparison table for the identical GMS query cohort.

Include at least:

| Configuration | Additional target training labels | Additional development labels | Outliers | NMAD | CE | CRPS |
|---|---:|---:|---:|---:|---:|---:|
| BS native historical head | 0 | 0 | measured | measured | measured | measured |
| BS old head, canonical input | 0 | 0 | measured | measured | measured | measured |
| BS fresh head, hard CE | 256 | 256 | measured | measured | measured | measured |
| BS fresh head, Gaussian sigma3 | 256 | 256 | measured | measured | measured | measured |
| B0 fresh head, Gaussian sigma3 | 256 | 256 | measured | measured | measured | measured |
| Photometric kNN | 256 | actual | measured | measured | measured | measured |
| BS-hidden kNN | 256 | actual | measured | measured | measured | measured |
| B0-hidden kNN | 256 | actual | measured | measured | measured | measured |
| BC-hidden kNN | 256 | actual | measured | measured | measured | measured |
| Constant-median baseline | 256 | 0 | measured | measured | not applicable | not applicable |

For methods without valid PDFs, report probability metrics as unavailable rather than making up a distribution.

Preserve the six-fit means and ranges for trainable heads.

Do not combine training-seed variation with galaxy bootstrap uncertainty.

Use galaxy-level paired bootstrap for matched prediction comparisons.

Produce an additional matrix showing which hypotheses are supported:

- Native-head loss.
- Serialization mismatch.
- Sparse classifier readout.
- Poor hidden-neighbourhood geometry.
- Easy narrow-redshift query distribution.
- Photometric feature advantage.
- Identity or feature leakage.

Assign verdicts such as:

- Supported.
- Partially supported.
- Not supported.
- Inconclusive.

Do not force a single explanation when multiple causes remain plausible.

---

# 4. Expected scientific figures

Prioritize a small number of informative plots.

### Figure 1 — Same-cohort model comparison

Compare the catastrophic-outlier fraction and NMAD for:

- Native BS.
- Hard-label fresh BS.
- Soft-label fresh BS.
- Photometric kNN.
- BS-hidden kNN.
- Constant-median baseline.

Use consistent query galaxies.

### Figure 2 — Hidden-space neighbourhood quality

Show local redshift differences versus nearest-neighbour distance for:

- Raw photometry.
- B0 hidden states.
- BS hidden states.
- BC hidden states.

Normalize each distance appropriately and do not imply that raw Euclidean distances across different spaces are directly comparable.

### Figure 3 — Redshift-conditioned failures

Show outlier rate versus true redshift for:

- Photometric kNN.
- BS native.
- BS fresh soft-label head.
- BS-hidden kNN.

Include support/query counts.

### Figure 4 — Readout learning and calibration

Compare hard-label and soft-label head learning curves, with selected checkpoint positions and CE/CRPS or coverage where valid.

Do not construct artificial smooth trends from only a few measurements.

---

# 5. Execution sequence

Complete the experiment in this order:

1. Verify the historical BS checkpoint and Stage 4 split/feature hashes.
2. Reproduce the original GMS kNN and BS hard-head results.
3. Run native BS original-head inference on the fixed query.
4. Run the sigma3 fresh-head comparison using the existing frozen features.
5. Run hidden-space kNN for B0, BS and BC.
6. Complete the kNN identity/feature and redshift-distribution audit.
7. Produce the joint hypothesis assessment and figures.

Use verified frozen caches rather than repeatedly evaluating all 4B backbones.

Only compute missing hidden states when necessary.

Do not start a new backbone training job.

If a verification step identifies an integrity problem, record it before proceeding and clearly distinguish invalid historical metrics from corrected results.

Avoid expensive sweeps or unrelated architectural experiments.

---

# 6. Deliverables

Use a new versioned directory, for example:

`stage4_5/readout_knn_discrepancy_1029/`

Save:

- `checkpoint_replay_and_native_head.md`
- `native_vs_canonical_metrics.csv`
- `soft_label_readout_metrics.csv`
- `hidden_knn_metrics.csv`
- `knn_leakage_and_features_audit.md`
- `constant_baseline_and_query_distribution.csv`
- `neighbourhood_diagnostics.csv`
- `paired_comparisons.csv`
- `scientific_assessment.md`
- The four requested figures.
- All experiment configurations and hashes.
- Reproduction commands.
- Runtime and memory summaries.

The final report should explicitly answer:

1. How well does the original BS model perform on the same GMS query objects?
2. How much does canonical serialization degrade the original BS head, if at all?
3. Does sigma3 smoothing repair the fresh-head performance gap?
4. Does kNN on BS hidden states recover more useful redshift information than the fresh BS head?
5. Are BS hidden-state neighbourhoods more redshift-informative than B0 neighbourhoods?
6. Why does direct photometric kNN achieve such a low outlier fraction?
7. Is that result reproducible and free from identified identity/feature leakage?
8. What portion of the apparent BS–kNN discrepancy can reasonably be attributed to readout, representation, data distribution and evaluation protocol?
9. Does Stage 4's BC→GMS positive representation-transfer result remain scientifically interpretable after these diagnostics?
10. What is the minimum next experiment needed for genuine CLAUDS few-shot generalisation?

**Execute all four checks and produce the integrated assessment. Do not stop after writing a plan.**

**The purpose is to understand the discrepancy, not to optimize the benchmark until Qwen wins.**