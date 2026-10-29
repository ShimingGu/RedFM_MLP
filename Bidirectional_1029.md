# Codex Execution — Bidirectional Matched Transfer and Seed-42 Matched-Training Pilot

## Objective

Execute the two experiments prepared in the completed October 28 Legacy checkpoint audit and matched-training package.

This is explicit authorisation to:

1. Run frozen bidirectional cross-survey inference for the four existing original Qwen models on the mutually independent paired cohort.
2. Launch the four fresh seed-42 matched-training models using the prevalidated experimental package.

**This authorisation does not include matched-P inference for newly trained models, seeds 43/44, additional hyperparameter searches, or modification of the filter-metadata counterfactual experiments.**

The scientific objective is to measure how the observational system used during post-training affects photo-z transferability, and whether this depends on QLoRA versus DoRA.

## Existing verified resources

The completed preparation package is:

`/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1028/matched-training-design`

Package manifest SHA256:

`bc7ac99fda7629f1235fb1a4c547e699fa9039984e82a474ebe654d418183661`

Original HSC matched benchmark:

`/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1027/hsc-legacy-paired-full`

The checkpoint audit has already established:

- Original HSC QLoRA and DoRA checkpoints are available and validated.
- Original Legacy QLoRA and DoRA checkpoints are now recovered, including their trained PDF heads.
- Both Legacy checkpoints reproduce 64 original validation objects exactly.
- The original Legacy training and validation IDs have been deterministically reconstructed and independently verified.
- The mutually independent original-model evaluation cohort contains **61,310 paired galaxies**.
- The fresh matched split contains **16,600 train / 4,257 validation / 61,696 locked evaluation** galaxies.
- The new training factory, Level-1 serializer, initialisation checks, partial-window updates and resume logic have already passed their preparation tests.

Do not repeat this entire audit unnecessarily.

Verify package hashes, manifests and required paths before execution. Reuse all validated artifacts.

---

# STAGE 1 — Complete original-model bidirectional transfer

## 1. Freeze the evaluation cohort

Use the exact existing `P_mutual` cohort of 61,310 physical galaxies.

This cohort excludes training and validation exposure for all seven original HSC/Legacy checkpoint families previously audited.

Maintain:

- Original galaxy identifiers and ordering.
- Original HSC and Legacy measurements.
- Shared DESI spectroscopic-redshift labels.
- The established spectroscopic quality cuts.
- The original independent-exposure exclusions.
- Existing 300-bin PDF grid over [0, 4.5].

Do not add back the 386 objects excluded from the original 61,696 P cohort.

Do not select objects based on prediction quality.

Save an immutable cohort identity manifest and its hash.

## 2. Recover the exact four original models

Use the completed original checkpoints:

**HSC source**

- Original HSC Qwen3.5-4B-Base QLoRA.
- Original HSC Qwen3.5-4B-Base DoRA.

Retrieve their exact selected states from the original HSC benchmark manifest.

**Legacy source**

QLoRA:

`/arc/home/gsm/aion_output/figures/qwen-desi-ls-qlora-specz-e10/qlora`

DoRA:

`/arc/home/gsm/aion_output/figures/qwen-desi-ls-dora-specz-e10/dora`

Load the verified selected adapter and corresponding trained `photoz_head.pt`.

Do not substitute rolling restart states or reinitialise the prediction heads.

Before full inference, replay the archived small reference inputs and compare logits/PDFs against the saved source-validation proofs.

Any unexplained mismatch is an implementation blocker.

## 3. Run the eight-cell historical benchmark

Evaluate:

| Trained model | HSC observations | Legacy observations |
|---|---|---|
| HSC original QLoRA | H-Q → H | H-Q → L |
| HSC original DoRA | H-D → H | H-D → L |
| Legacy original QLoRA | L-Q → H | L-Q → L |
| Legacy original DoRA | L-D → H | L-D → L |

All eight cells must use the same 61,310 galaxies and the same spectroscopic labels.

Use each original model's trained serializer contract.

### Critical: cross-domain input validity

The original Level-0 models may have different source-specific feature schemas.

Before scoring cross-domain predictions, inspect actual serialized records.

Never silently substitute:

- HSC i or y measurements for Legacy W1/W2.
- Legacy W1/W2 for HSC i/y.
- Measurements from different physical filters merely because they occupy the same array position.

Where the original serializer supports explicit missing values, preserve unsupported source-specific fields as missing rather than inventing photometry.

Where a physically faithful mapping is impossible without changing the trained input semantics, explicitly identify the condition as an **interface-stress diagnostic**, not a clean filter-aware observational transfer.

Do not modify weights or retrain the historical model to resolve a schema incompatibility.

The historical eight-cell comparison is important, but scientifically valid input interpretation takes priority over filling every cell.

The fresh Level-1 matched-training experiment in Stage 2 is intended to provide the cleaner shared-interface comparison.

## 4. Reuse verified historical predictions where possible

The archived HSC benchmark already contains full-cohort predictions for the original HSC-source QLoRA and DoRA models.

For any reused prediction:

1. Verify model checkpoint hash.
2. Verify input serializer and observational representation.
3. Verify PDF grid and probability-mass convention.
4. Verify exact galaxy identifiers.
5. Slice and reorder to `P_mutual`.
6. Confirm identical predictions on a small live replay.

Reuse is preferable to expensive duplicate inference when equivalence is established.

Never substitute metrics from 61,696 galaxies for metrics from 61,310 galaxies.

All final Stage-1 comparisons must be recomputed on the same `P_mutual` IDs.

## 5. Metrics and transfer comparisons

For every valid cell, report:

- Cross-entropy (CE).
- CRPS.
- NMAD.
- Catastrophic-outlier fraction, with |delta| > 0.15.
- Mean and median redshift bias.
- Nominal 68% coverage.
- PIT diagnostics.
- Predicted versus spectroscopic redshift.
- Object-level normalised residuals.

Use:

\[
\delta_i =
\frac{z_{{\rm phot},i}-z_{{\rm spec},i}}
{1+z_{{\rm spec},i}}.
\]

Calculate same-model transfer penalties:

\[
P_{H\to L}
=
\mathrm{CE}(M_H,X_L)-\mathrm{CE}(M_H,X_H),
\]

\[
P_{L\to H}
=
\mathrm{CE}(M_L,X_H)-\mathrm{CE}(M_L,X_L).
\]

Calculate target-native gaps:

\[
G_{H\to L}
=
\mathrm{CE}(M_H,X_L)-\mathrm{CE}(M_L,X_L),
\]

\[
G_{L\to H}
=
\mathrm{CE}(M_L,X_H)-\mathrm{CE}(M_H,X_H).
\]

Evaluate QLoRA versus DoRA separately for each transfer direction.

Include absolute native/target CE so that a smaller transfer penalty cannot be misinterpreted as stronger transfer when it merely reflects degraded source performance.

## 6. Statistical and redshift analyses

Use the existing predeclared redshift boundaries:

`[0.0, 0.2, 0.4, 0.6, 1.0, 4.5]`

Produce:

- Full-cohort results.
- Per-bin results.
- Native-population-weighted results.
- Equal-bin macro summaries.
- Paired galaxy-level differences.
- Newly catastrophic and rescued outlier counts.
- 1,000 shared paired-bootstrap draws and 95% confidence intervals.

Use the same bootstrap indices across all four models and both observation systems.

Preserve the distinction between population uncertainty conditional on the trained models and unmeasured training-seed uncertainty.

The original paired sample has little representation above z=1 and no objects above z=2. Do not claim strict high-z extrapolation.

## 7. Stage-1 output

Create a non-overwriting output directory under the existing matched-training project, for example:

`/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1028/original-bidirectional-transfer`

Required outputs:

- `report.md`
- `results.json`
- `manifest.json`
- Cohort and exposure hashes.
- Checkpoint replay proofs.
- All eight valid inference-cell metrics.
- Model/input compatibility audit.
- Object-level predictions and paired differences.
- Bootstrap arrays.
- Redshift-stratified results.
- Prediction, residual, calibration and transfer-penalty plots.

Report which cells were genuinely completed, which reused verified predictions, and which—if any—could not be interpreted faithfully because of original source-specific serializer limitations.

**Complete Stage 1 before starting Stage 2.**

However, the Stage-1 scientific results must not be used to modify the already-predeclared Stage-2 training protocol.

---

# STAGE 2 — Launch fresh matched-training seed-42 pilot

## 8. Authorised training matrix

Launch exactly four new models:

| Training system | QLoRA | DoRA |
|---|---|---|
| HSC grizy | H-Q-42 | H-D-42 |
| Legacy grzW1W2 | L-Q-42 | L-D-42 |

Use seed **42** only.

All four must start from the same verified pre-adaptation:

`/arc/home/gsm/hf_models/Qwen3.5-4B-Base`

Do not initialise from previously trained HSC or Legacy adapters.

Use the saved canonical seed-42 initial prediction head.

Use the verified common Level-1 serializer, with correct physical survey, filter and instrument identities.

Do not add `pivot_nm`, `width_nm` or other Level-2 metadata.

The training galaxies and redshift labels are identical between HSC and Legacy branches.

## 9. Lock the matched dataset

Use the existing prepared split:

| Partition | Number of galaxies |
|---|---:|
| Train | 16,600 |
| Validation | 4,257 |
| Locked P evaluation | 61,696 |

Train and validate only on the first two partitions.

The locked P dataset must not enter the new-model training process.

Do not change the sample selection, redshift distribution, photometric quality cuts, ordering rules or source-input representation based on Stage-1 results.

Use the existing saved training permutations.

## 10. Training protocol

Execute the prepared seed-42 configuration without introducing new hyperparameter searches.

Per model:

- Ten fixed epochs.
- Three head-only epochs.
- Seven joint head-and-adapter epochs.
- Effective batch size 16.
- Microbatch size 1.
- Gradient accumulation 16.
- 16,600 presentations per epoch.
- 1,038 optimizer updates per epoch.
- **10,380 total optimizer updates.**
- **166,000 total training presentations.**

Use the verified eight-example final accumulation window at each epoch boundary.

Optimisation:

- AdamW.
- Head LR: 2e-4.
- Adapter LR: 1e-5 during joint training.
- Adapter LR: zero during head-only training.
- Head weight decay: 0.01.
- Adapter weight decay: zero.
- Separate head/adapter gradient clipping, 1.0 and 0.1.
- Linear decay with 3% warmup over the respective active budgets.

Maintain the prepared QLoRA/DoRA rank-8 configuration, alpha, dropout and target modules.

Use the identical saved head initialisation for all four branches.

Use the prepared seed streams for adapter initialisation, presentation order and dropout.

Do not force identical random trajectories between different architectures when their random-number consumption differs.

## 11. Source-validation and checkpoint selection

Evaluate each model on its own observational system at every epoch end.

Select the best checkpoint by minimum source-validation CE.

Record at every epoch:

- Training loss.
- Validation CE.
- Validation NMAD.
- Validation catastrophic-outlier fraction.
- CRPS.
- Bias.
- 68% coverage.
- Gradient/update diagnostics.
- Adapter and head parameter changes.

Train all four models to the same fixed ten-epoch budget, regardless of which epoch is selected as best.

Save both:

1. The selected best-source-validation checkpoint.
2. The final fixed-budget checkpoint.

Preserve their adapters, heads, exact configurations, optimiser states and training histories.

Do not silently use different early-stopping budgets for HSC and Legacy.

### Convergence diagnostics

Fresh Level-1-from-base training has not previously been shown to converge under this exact protocol.

Monitor whether all branches learn nontrivial redshift inference.

Flag:

- Validation CE remaining close to an uninformative baseline.
- Nearly constant redshift predictions.
- Exploding or vanishing gradients.
- Nonfinite loss.
- Adapter inactivity.
- Major deterioration after head-only warmup.
- Systematic source-domain underfitting.
- Checkpoint-selection anomalies.

Do not interpret poor cross-survey generalisation if the underlying model has not learned a useful source-domain mapping.

If a branch cannot make meaningful training progress, investigate implementation failures before concluding anything about model capacity.

Do not change learning rates, training epochs or model architecture in only one branch to rescue its performance.

If an actual protocol change is necessary, report the issue and propose a matched revision for all four branches rather than silently changing the experiment.

## 12. GPU scheduling and resource protection

The preparation audit identified three currently available MIG slices and existing CLAUDS jobs on other slices.

Use the verified resource-reservation and advisory-locking functionality provided by the launcher.

Before execution:

- Recheck which MIG slices are actually available.
- Do not assume that a slice remains free because it was free during the previous audit.
- Do not displace existing CLAUDS or other reserved jobs.
- Verify memory feasibility for the first actual backward pass.
- Record measured memory and throughput.
- Preserve resumable progress in the event of interruption.

The projected four-model training/validation cost is approximately 182 MIG-GPU hours, based on historical native-input proxies.

This is not a requirement to meet a particular runtime.

Run only on safely available resources.

Use the prepared launch command:

```bash
PIXI_CACHE_DIR=/scratch/.tmp-gsm/pixi-cache \
pixi run --locked python \
notebooks/launch_matched_training_1028.py \
--package \
/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1028/matched-training-design \
--seeds 42
```

Verify that the prepared launcher still matches the audited package and that its output location cannot overwrite existing runs.

Use its established resume and completion semantics.

## 13. Hard boundary: no new-model P scoring

After completing seed-42 training:

**Do not run any of the new H-Q-42, H-D-42, L-Q-42 or L-D-42 models on the locked 61,696-galaxy P cohort.**

Specifically:

- No P cross-entropy.
- No P NMAD.
- No P predicted-redshift diagnostic.
- No exploratory P subset.
- No selected-versus-final P comparison.
- No target-based calibration.
- No choosing training budgets or future seeds based on new-model transfer performance.

The reason is methodological.

We want to inspect the source-validation convergence of the four fresh models before deciding whether a fixed-budget revision or seeds 43/44 are necessary.

If all four converge adequately, the next separate authorisation can freeze the final evaluation procedure and permit the matched-P matrix.

The historical original-model Stage-1 inference is separately authorised and does not override this restriction.

## 14. Do not launch additional seeds

Do not start seeds 43 or 44.

Do not automatically expand the pilot after seed 42 finishes.

Their configurations must remain preserved and ready for later execution.

A future multi-seed decision should depend on source-validation convergence, computational feasibility and method consistency—not on a favourable cross-domain P result.

## 15. Stage-2 completion checks

Verify for every model:

- Exactly 10,380 optimizer updates.
- Exactly 166,000 training presentations.
- Complete ten-epoch source-validation history.
- Matching source-galaxy identities and labels.
- Correct head-only/joint transition.
- Correct model and head initialisations.
- Correct QLoRA/DoRA adapter parameter coverage.
- Selected checkpoint exported.
- Fixed-final checkpoint exported.
- Original training and validation prediction outputs saved.
- Optimiser/scheduler/RNG resume state retained.
- Source-validation checkpoint replay passes.
- No locked-P inference occurred.

If a model is interrupted, resume from its own verified state rather than restarting with a new uncontrolled initialisation.

---

# Final reporting

Produce one top-level consolidated report covering both stages.

## Section A — Historical bidirectional transfer

- Four original models.
- Eight inference cells.
- Exact 61,310-object cohort.
- Source-domain and cross-domain scores.
- Bidirectional transfer penalties.
- Target-native gaps.
- QLoRA/DoRA comparisons.
- Redshift dependence.
- Paired confidence intervals.
- Source-specific interface limitations.
- Scientific findings and unresolved confounders.

## Section B — Seed-42 matched training

- Four fresh models.
- Verified matched source inputs and splits.
- Checkpoint initialisation proofs.
- Ten-epoch training histories.
- Source-validation metrics.
- Selected versus final checkpoint information.
- Actual GPU cost, memory and throughput.
- Model-training convergence assessment.
- Any implementation or optimisation issues.

## Section C — Recommendation for the next gate

Answer:

1. Are the original Legacy → HSC results scientifically interpretable, or partly limited by native Level-0 interface incompatibilities?
2. Does the original bidirectional benchmark show directional transfer asymmetry?
3. Do original QLoRA and DoRA have consistently different transfer behaviour?
4. Have all four fresh Level-1 seed-42 models learned useful source-domain redshift mappings?
5. Is the fixed ten-epoch protocol adequate for both source systems?
6. Should the exact current protocol be extended to seeds 43/44, based only on source-validation evidence and resource feasibility?
7. Are the newly selected/final checkpoint pairs ready for subsequent locked-P evaluation?

Preserve all original models, archived benchmarks, filter metadata, the October 28 preparation package and unrelated experiments.

**Execute both authorised stages. Do not stop at another design proposal.**

The required end state is:

- Completed historical bidirectional matched-transfer results.
- Completed fresh seed-42 matched training of four models.
- A source-validation convergence report.
- Preserved, unscored locked-P evaluation configurations for the next scientific decision.