# Experiment — Filter-Metadata Counterfactuals for HSC → Legacy Transfer

## Scientific objective

We have completed a large, same-galaxy benchmark of photometric-redshift transfer from HSC grizy to DESI–Legacy grzW1W2 using post-trained Qwen3.5-4B-Base.

The benchmark includes 61,696 independent paired galaxies, each observed in both photometric systems, with identical DESI spectroscopic redshift labels.

Our Level-2 QLoRA model, which receives explicit filter identities, pivot wavelengths and equivalent bandwidths, shows substantially better cross-survey transfer than the matched Level-1 model.

However, we do not know whether this improvement reflects:

1. Physically meaningful use of filter wavelengths and bandwidths.
2. Sensitivity to arbitrary numerical metadata.
3. A favourable but nonphysical change in tokenisation or attention.
4. General adaptation to longer input prompts.
5. Other representation-dependent effects.

**Your task is to test the causal sensitivity of the existing trained model to the numerical values of filter metadata, without performing additional training.**

This experiment must distinguish model sensitivity to metadata from evidence of physically meaningful generalisation.

Do not presume that the model understands filter physics.

---

## 1. Existing benchmark and checkpoints

Locate the completed benchmark at:

`/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1027/hsc-legacy-paired-full`

Inspect its:

- `manifest.json`
- `evaluation_policy.json`
- `cohort.npz`
- `eligible_pairs.npz`
- `results.json`
- Existing serialized input records
- Benchmark evaluation and plotting code

The benchmark's independent P cohort contains 61,696 galaxies. The original HSC training and validation objects were excluded.

Five HSC-trained checkpoints were evaluated, but this experiment primarily concerns:

- **B:** HSC warm-start Level-1 QLoRA, with explicit filter identities.
- **C:** HSC warm-start Level-2 QLoRA, adding pivot wavelength and rectangular equivalent bandwidth.

The reported independent-cohort metrics are:

| Model | HSC CE | Legacy CE | Transfer ΔCE | Legacy NMAD |
|---|---:|---:|---:|---:|
| B | 2.5245 | 3.1430 | 0.6186 | 0.0483 |
| C | 2.5672 | 3.0065 | 0.4394 | 0.0475 |

These are reference values that must be reproduced before counterfactual inference.

The observed C-minus-B difference in transfer penalty is approximately −0.1792.

All models use the same 300-bin redshift grid over 0–4.5.

Use the exact saved checkpoint identities and hashes. Do not infer paths from model names.

## 2. Hard constraints

This is a **frozen-weight counterfactual experiment**.

- No additional QLoRA, DoRA, SFT or other training.
- No modifications to model weights or PDF heads.
- No target-label fitting or recalibration.
- No target-informed checkpoint or perturbation selection.
- No replacement of existing benchmark artifacts.
- No changes to galaxy membership, order or spectroscopic labels.
- Preserve photometric fluxes, uncertainties, validity masks and EBV.
- Preserve actual filter identities and survey/instrument identifiers unless a condition explicitly requires otherwise.
- Do not evaluate unrelated held-out test datasets.
- Do not treat the exploratory V cohort as independent evidence.

The existing 61,696-object P cohort is the primary benchmark.

The original redshift labels may be used for final evaluation, but not for selecting or constructing perturbations.

## 3. Counterfactual design

Implement a configurable metadata intervention layer on top of the existing Level-2 serializer.

The intervention must modify only:

- `pivot_nm`
- `width_nm`

All other fields must remain unchanged.

The correct baseline uses the immutable existing filter registry:

`configs/filter_registry_v1.json`

Do not edit or overwrite this file.

Store all counterfactual metadata mappings separately.

### Condition 0 — Correct metadata

Use the original Level-2 serializer and correct physical metadata.

This is the unmodified control.

Verify exact reproduction of the archived C predictions on both HSC and Legacy measurements.

### Condition 1 — Paired metadata permutation

Keep each filter name and observed flux unchanged, but reassign its wavelength and width using a fixed permutation of the available filter metadata.

Permute the `(pivot_nm, width_nm)` pairs together.

For example, a filter named `HSC_g` retains its name and measured flux, but receives another filter's wavelength and bandwidth.

The permutation must:

- Be generated independently of galaxy photometry and redshift labels.
- Be fixed across all galaxies.
- Be reproducible from a recorded seed.
- Preserve the original collection of numerical metadata pairs within the relevant permutation group.
- Prefer derangements, so no filter retains its original metadata when feasible.
- Preserve units and formatting.

Generate several predetermined permutations, including a cyclic permutation and a seeded derangement.

Do not run many permutations and report only the one producing the largest effect.

Respect the actual survey/instrument-dependent Legacy filter identifiers. Construct each mapping at the physical filter-profile level rather than incorrectly assuming that all Legacy g/r/z observations use identical filters.

### Condition 2 — Wavelength-only permutation

Keep:

- Filter names unchanged.
- Fluxes and uncertainties unchanged.
- Equivalent bandwidths correct.

Permute only the pivot wavelengths among physical filters.

Use predetermined, globally fixed mappings.

This tests sensitivity specifically to wavelength information.

### Condition 3 — Bandwidth-only permutation

Keep:

- Filter names unchanged.
- Pivot wavelengths correct.
- Fluxes and uncertainties unchanged.

Permute only the equivalent bandwidths.

This tests sensitivity specifically to width information.

### Condition 4 — Controlled wavelength perturbation

Apply predetermined small fractional perturbations to the numerical pivot wavelengths.

Use, for example:

- +2%
- −2%
- +5%
- −5%

These are counterfactual inputs, not alternative estimates of the actual filter profiles.

Preserve the correct bandwidth values and filter names.

Use identical perturbation rules across all galaxies, and include both positive and negative directions.

The purpose is to examine the local response to physically interpretable changes in wavelength.

Avoid constructing the perturbations based on target prediction errors.

### Condition 5 — Matched numerical placebo

Construct a Level-2-like input containing numerical metadata with the same field structure, units, formatting and approximately matched numerical/token-length distributions, but with no correct association between filter identity and the physical response parameters.

Prefer fixed numerical assignments drawn or permuted from the verified metadata registry over arbitrary extreme constants.

The placebo must remain fixed for each physical filter across all galaxies.

Document exactly how the placebo values were generated.

A placebo cannot automatically isolate tokenisation effects merely because it has the same number of fields. Audit actual token IDs and token counts.

---

## 4. Preserve experimental identifiability

There is an important complication.

The Level-2 model was trained on HSC with the correct filter metadata, which remained effectively constant for each HSC band across all training objects.

Consequently, changing metadata at inference time creates a distribution shift in the textual interface.

A performance degradation after metadata corruption does **not**, by itself, prove that the model understands physical filter response.

It may simply indicate sensitivity to unexpected numerical tokens.

To help distinguish these possibilities, perform the counterfactuals on both observational representations of the same galaxies:

1. HSC grizy observations, the source observational system.
2. Legacy grzW1W2 observations, the transferred observational system.

Apply consistent intervention rules independently to each system's real filter identities.

For every condition, report:

- Effect on HSC predictions.
- Effect on Legacy predictions.
- Effect on the matched observational transfer penalty.

For target-only intervention experiments, keep the HSC input unchanged and explicitly label the resulting transfer-penalty difference as a target-side perturbation effect.

For both-domain intervention experiments, manipulate both HSC and Legacy metadata consistently with the documented rule.

Do not conflate these two designs.

## 5. Predeclare the evaluation plan

Before scoring any new counterfactual predictions:

1. Freeze the list of experimental conditions.
2. Freeze the metadata mappings and random seeds.
3. Freeze all target galaxy identifiers.
4. Freeze the evaluation metrics and redshift-bin boundaries.
5. Save a versioned evaluation-policy file with a SHA-256 hash.

Use the existing redshift bins:

`[0.0, 0.2, 0.4, 0.6, 1.0, 4.5]`

The high-redshift tail contains only approximately 400 objects above z=1. Treat its estimates separately and do not make strong claims about z>2 generalisation.

For computational efficiency, an initial inference smoke test may use a predetermined, label-blind hash-selected subset.

Use that subset only for checking implementation and computational feasibility, not for selecting the most favourable metadata condition.

Evaluate the complete predeclared primary condition set on all 61,696 P galaxies if resources permit.

If resource constraints prevent completing the full matrix, report exactly which predetermined conditions were run. Do not choose conditions according to preliminary redshift metrics.

## 6. Serialization and tokenisation audit

For every counterfactual condition, inspect actual final tokenizer inputs.

Verify:

- Photometry remains byte-identical to the original C condition.
- Filter names remain unchanged.
- Only the intended physical metadata fields differ.
- Wavelengths and bandwidths remain in nanometres.
- Numeric precision matches the original Level-2 serializer.
- No missing measurements are changed.
- Source and target objects remain in identical order.
- No records are silently truncated.

Report:

- Token count distributions.
- Maximum sequence length.
- Frequency of changed token IDs.
- Any systematic token-length differences between counterfactual conditions.

If the original context limit cannot accommodate every variant, establish one common adequate limit before scoring, verify baseline replay, and use the same limit across conditions.

Do not claim a length-matched control unless the actual tokenised lengths have been verified.

## 7. Evaluation metrics

For every completed condition and observational system, compute:

- CE.
- CRPS.
- NMAD.
- Catastrophic-outlier fraction.
- Mean and median normalised redshift bias.
- Nominal 68% coverage.
- Predicted-redshift distributions.
- PIT diagnostics where already supported.

The primary metric is CE, because the model predicts a redshift probability distribution.

NMAD and catastrophic outliers remain essential complementary metrics.

### Primary counterfactual effect

For condition k, define:

\[
\Delta_{\mathrm{meta}}^L(k)
=
\mathrm{CE}(C_k,X_L)
-
\mathrm{CE}(C_{\mathrm{correct}},X_L).
\]

This directly measures how modifying filter metadata changes Legacy predictive performance under fixed model weights.

Calculate the analogous HSC quantity:

\[
\Delta_{\mathrm{meta}}^H(k)
=
\mathrm{CE}(C_k,X_H)
-
\mathrm{CE}(C_{\mathrm{correct}},X_H).
\]

These expressions denote the same trained C model evaluated with different serialized metadata, not differently trained checkpoints.

For conditions in which both domains are perturbed, also calculate:

\[
\Delta\Delta_{\mathrm{meta}}(k)
=
\Delta_{\mathrm{meta}}^L(k)
-
\Delta_{\mathrm{meta}}^H(k).
\]

Interpret this as the difference in metadata sensitivity between domains.

Do not automatically interpret it as an improvement or degradation of physical understanding.

### Comparison with Level 1

Keep the original frozen Level-1 model B as an external reference.

Compare:

- C with correct metadata.
- C with counterfactual metadata.
- B with no physical metadata.

Remember that B and C were independently adapted from the same archived HSC checkpoint and have different final weights.

Therefore, changing metadata in C isolates inference-time sensitivity, whereas C versus B measures the effect of the broader training/representation intervention.

These are distinct comparisons.

## 8. Statistical analysis

Use paired-galaxy bootstrap resampling.

- At least 1,000 predetermined bootstrap samples for the principal comparisons.
- Reuse identical bootstrap indices across model and metadata conditions.
- Report percentile 95% intervals.
- Include differences in CE and CRPS, not merely separate uncertainties for each model.
- Report the fraction of galaxies whose absolute normalised redshift residual improves or worsens.
- Report newly catastrophic and rescued outliers.

All confidence intervals are conditional on the fixed trained checkpoint and the evaluated galaxy population.

They do not measure training-seed uncertainty or establish reproducibility across post-training runs.

For multiple metadata permutations, show the entire distribution of outcomes rather than only the most extreme perturbation.

## 9. Redshift-dependent metadata sensitivity

Produce a stratified analysis for each predetermined redshift bin.

At minimum, compare:

1. Correct Level-2 metadata.
2. Paired wavelength/width permutation.
3. Wavelength-only permutation.
4. Bandwidth-only permutation.
5. Matched numerical placebo.

Report absolute HSC and Legacy predictive quality as well as the metadata-induced changes.

Particular attention should be paid to the existing non-uniform behaviour:

- Level 2 performs worse than Level 1 in some redshift intervals.
- Level 2 performs substantially better in others.
- Its aggregate advantage depends on the evaluation redshift distribution.

Determine whether the sensitivity to incorrect filter metadata also varies with redshift.

Do not treat redshift-bin variation alone as evidence of physical spectral-feature reasoning.

## 10. Scientific interpretation

The final report must explicitly address the following hypotheses.

### H1 — Metadata values are effectively ignored

If modifying physical metadata produces negligible changes in predictions, the existing Level-2 benefit may arise from formatting, training dynamics or other representation effects rather than the specific numerical values.

### H2 — The model uses metadata numerically but not physically

If changing metadata substantially modifies predictions, but physical correctness does not consistently outperform plausible incorrect assignments, then the result supports numerical/token sensitivity, not demonstrated physical understanding.

### H3 — Correct physical metadata is systematically advantageous

If the physically correct assignment consistently outperforms predetermined incorrect assignments, especially on Legacy while preserving source performance, this supports the hypothesis that the model's predictions benefit from correct filter metadata.

However, this is not sufficient to establish a learned physical response model.

### H4 — Physically structured sensitivity

If modest wavelength perturbations produce repeatable, interpretable changes in predicted redshifts or probability distributions, this could motivate a further investigation of physically meaningful filter-response reasoning.

Such patterns must be interpreted cautiously because a change in the filter token sequence need not correspond to a physically consistent change in the observed galaxy spectrum.

Do not claim that the model has learned the redshifting of spectral features solely from these perturbations.

## 11. No new training in this experiment

Do not retrain C using shuffled metadata.

Such an experiment could be scientifically important, but it answers a different question:

**Would correct physical metadata outperform incorrect metadata under matched training conditions?**

That would require separate models trained from the same initial checkpoint with matched optimisation, seeds and source-validation procedures.

The present experiment instead asks:

**Does the already trained Level-2 model actually depend on the supplied metadata values, and does physical correctness matter at inference time?**

Keep those questions separate.

If necessary, recommend a later matched-training intervention, but do not launch it without separate authorisation.

## 12. Deliverables

Create a new non-overwriting experiment directory under the existing project structure, for example:

`filter-aware-runs/1027/metadata-counterfactual/`

or the corresponding persistent project output location.

Deliver:

1. A verified baseline replay of original B and C predictions.
2. A versioned counterfactual metadata mapping and evaluation policy.
3. Serializer tests confirming only intended fields change.
4. Tokenisation and truncation audits.
5. A full primary-cohort table of HSC and Legacy results.
6. Paired counterfactual effects with bootstrap intervals.
7. Redshift-stratified results.
8. Plots comparing predicted PDFs, redshift residuals and metadata sensitivity.
9. Machine-readable per-object predictions and metrics.
10. A concise scientific report separating established findings, tentative interpretations and unresolved mechanisms.
11. Reproducible commands and checkpoint/data hashes.

Do not overwrite or modify existing checkpoints, filters, benchmark results or evaluation manifests.

Do not launch additional SFT.

**Final scientific question:**

Does the improved HSC-to-Legacy transfer of the Level-2 post-trained Qwen model depend on physically correct filter wavelengths and bandwidths, or can the same behaviour arise from arbitrary numerical metadata and altered text representation?

Implement and execute the experiment rather than stopping after a design proposal.