# Stage 3 — Real-Survey Readiness and Pre-Registered Physics-Informed Cross-Survey Photo-z

## Overall scientific objective

We have completed a sequence of Stage 0–2B.6 experiments investigating cross-survey photometric-redshift generalisation, physical filter representations, SED identifiability, and foundation-model pretraining.

The next objective is to move from primarily synthetic and small paired-source diagnostics toward a scientifically interpretable **real-survey cross-domain experiment**.

The central question is:

**Can a physics-informed numerical representation improve real cross-survey photo-z inference beyond an ordinary filter-aware neural model, under controlled source supervision and matched observational information?**

The principal transfer target remains CLAUDS.

We must distinguish:

1. HSC-to-Legacy generalisation.
2. Legacy-to-HSC generalisation.
3. Mixed spectroscopic-source training.
4. Transfer from spectroscopic source surveys to CLAUDS.
5. Transfer to redshift and photometric regimes inadequately represented in source training.

These are distinct experiments and must not be conflated.

This project will run in two stages:

- **Stage 3A:** Dataset recovery, coverage audit, target spectroscopy and benchmark pre-registration.
- **Stage 3B:** Controlled real-survey N1 versus N1_kernel training and evaluation.

Finish Stage 3A before launching Stage 3B.

Do not start pretrained-Qwen training or Stage 2C.

---

# Part I — Stage 3A: Data Recovery and Real-Survey Readiness

## 1. Read the experimental history

Inspect the complete reports, manifests and relevant implementation code from:

- Stage 0 and Stage 0.5.
- Stage 1.
- Stage 2A and 2A.5 and 2A.6.
- Stage 2B and 2B.5 and 2B.6.

Pay particular attention to:

- Existing source/target photometric interfaces.
- Spectroscopic redshift provenance.
- Existing source train/validation/test assignments.
- Catalogue identity and crossmatch audits.
- The verified 21-filter union-knot response integrator.
- The numerical signed-flux and uncertainty-mask interface.
- N1 shared-set encoder.
- N1_kernel full basis-response representation.
- Physical NNLS/template baselines.
- Archived checkpoint and data hashes.

Do not reproduce all previous experiments.

Reuse verified components and retain their limitations.

## 2. Scientific findings motivating Stage 3

Previous work established:

### Foundation-model limitations

The original HSC-trained Qwen model achieves in-domain NMAD approximately 0.0268, but fails catastrophically on the CLAUDS diagnostic sample.

Pretrained Qwen representations have not demonstrated a robust, general cross-filter advantage over matched numerical controls.

Strong sensitivity to numerical serialization, final-token identity and absolute brightness was measured.

No astrophysical-pretraining-specific benefit has been established.

### Physical inference

The original unrestricted 12-template physical fitting model fails on HSC grizy because a flexible SED mixture can obtain almost exact fits at many incorrect redshifts.

The true-redshift SED fit is frequently incompatible with HSC's very small formal measurement errors.

Adding real WISE W1 measurements substantially improves redshift discrimination on matched spectroscopic galaxies.

Reduced SED flexibility can improve point redshifts but can also destroy true-redshift fit adequacy.

Marginalisation does not repair a physically incorrect SED family.

### Neural physics integration

The N2 physical-score approach exhibits conditional benefits but substantial sensitivity to spectral mismatch, score temperature, template error and training seed.

Its physics-trust gates have not shown stable cross-filter robustness and calibration.

More importantly, the N1_kernel control, which directly receives redshift-dependent physical basis-response information, substantially improves some synthetic held-response results.

For example, in Stage 2B.6, seed 11:

- Ordinary N1 M1 full-response outliers: 63.28%.
- N1_kernel M1 full-response outliers: 16.41%.
- Ordinary N1 M3 full-response outliers: 77.34%.
- N1_kernel M3 full-response outliers: 24.22%.

However, N1_kernel has additional parameters and access to external FSPS spectral information.

These experiments do **not** isolate the effect of architecture alone.

The next experiment must explicitly control these differences.

## 3. Recover the independent DESI–Legacy training population

The previous Stage 2B real-data pilot had access to an audited same-galaxy HSC+Legacy cohort, but not the complete independent Legacy training population.

The available Legacy validation cohort contains approximately 15,053 objects.

The previous paired-source pilot used approximately 1,024 unique training galaxies and 512 validation galaxies, with two survey views each.

This is not sufficient to characterise population-scale generalisation.

### Required work

Search the existing repository, caches, data manifests and accessible archival data for the original independent DESI–Legacy training catalogue.

The intended photometry includes:

- Legacy Survey DR9 Tractor FLUX_G/R/Z/W1/W2.
- Corresponding IVAR measurements.
- Galactic extinction metadata where available.
- Survey release/filter-system identification.
- DESI spectroscopic redshifts and reliable object identifiers.

Use existing project metadata and public archival documentation to verify exact catalogue versions.

Do not assume a file is the correct training population merely because its filename contains "Legacy".

If necessary, reconstruct the source training table from verified DESI spectroscopy and Legacy photometry using reproducible crossmatching and cuts.

Record:

- Data provenance.
- Release and selection criteria.
- Catalogue joins.
- Spectroscopic quality filters.
- Photometric quality filters.
- Training/validation/test assignments.
- Object identifiers.
- Source-row counts.
- Cryptographic hashes.
- Crossmatch ambiguities.

Do not silently replace the historical Legacy training set with a different selected population.

If the exact historical population is unrecoverable, construct a separately named new cohort and disclose all selection differences.

If downloads or remote data access are unavailable, complete the local audit and prepare exact reproducible acquisition commands rather than fabricating the dataset.

## 4. Audit the actual spectroscopic-redshift support

This is a mandatory deliverable.

For each real source dataset separately and for their deduplicated union, measure the distribution of:

- Spectroscopic redshift.
- Apparent magnitude.
- Flux.
- S/N.
- Photometric colours.
- Uncertainties.
- Available bands.
- Source selection and quality.
- Sky region.

Use the actual eligible source-training population, not the much smaller historical paired pilot, when it becomes available.

Report counts and fractions in at least:

- 0 <= z < 0.5.
- 0.5 <= z < 1.
- 1 <= z < 1.5.
- 1.5 <= z < 2.
- 2 <= z < 3.
- 3 <= z <= 4.5.

The earlier paired training pilot contained no z >= 2 objects.

Do not generalise that fact to the complete recovered DESI population without measuring it.

Also examine joint redshift–magnitude, redshift–S/N and redshift–colour coverage.

Estimate how much of CLAUDS lies outside source photometric support.

Reuse Stage 0's distance/domain-classifier diagnostics where the feature spaces and conventions remain comparable.

Distinguish:

- Lack of spectroscopic label support.
- Lack of photometric feature support.
- Missing wavelength coverage.
- Different measurement noise.
- Different galaxy populations.

These are not interchangeable concepts.

## 5. Audit source and target identity overlap

Use persistent catalogue identifiers wherever available.

For positional crossmatches:

- Apply documented coordinate conventions and matching radii.
- Report separations and multiple-match ambiguity.
- Avoid blindly treating every nearby match as the same galaxy.
- Include close/blended-object ambiguities in the audit.
- Ensure repeated HSC/Legacy views of the same galaxy remain in the same partition.

If a galaxy appears in both spectroscopic source training and a prospective CLAUDS target confirmation set, exclude it from source training and model selection.

Preserve separate reporting of intentionally matched HSC/Legacy galaxies.

Avoid leakage through:

- Duplicate photometric rows.
- Shared spectroscopic IDs.
- Multiple imaging measurements.
- Cross-catalogue spatial matches.
- Repeated observations.
- Source/test assignment collisions.

Where possible, additionally construct spatially separated confirmation partitions to test field-level dependence.

## 6. Recover independent CLAUDS spectroscopy

This is the most important readiness question.

The existing CLAUDS redshift reference is ZPHOT from a Phosphoros-based photometric catalogue.

It is not independent spectroscopic truth.

Search accessible project data and documented external spectroscopic catalogues for reliable matches to the CLAUDS photometric objects.

Potential sources may include appropriately verified public spectroscopy in the actual CLAUDS fields; do not assume any named spectroscopic programme covers the local catalogue.

Require:

- High-confidence spectroscopic redshifts.
- Documented spectroscopic quality criteria.
- Verified photometric–spectroscopic association.
- Explicit duplicate and ambiguous-match treatment.
- Source-training overlap exclusions.
- A saved provenance and quality manifest.

The spectroscopic confirmation cohort must be reserved for evaluation only.

Do not use its redshift labels to select model architecture, SED bases, loss functions, training steps, checkpoint, calibration, redshift prior or augmentation.

Where practical, quarantine confirmation labels until the experiment protocol and candidate models are frozen.

If an independent spectroscopic target cohort cannot be obtained, do not claim spectroscopic CLAUDS transfer.

A ZPHOT-only evaluation may be performed as a separately labelled exploratory agreement test, but must never be substituted for independent truth.

## 7. Establish the evaluation cohorts

Create clearly named, nonoverlapping cohorts.

### S1 — HSC spectroscopic source

GalaxiesML-Spec, with native verified HSC grizy photometry and DESI spec-z.

### S2 — Legacy spectroscopic source

Recovered independent Legacy photometry with DESI spec-z.

### S3 — Same-galaxy paired HSC/Legacy cohort

Use for controlled multi-view and observational-system comparisons.

Do not describe S3 as an independent mixed-source population.

### T1 — Existing CLAUDS exploratory cohort

The previously inspected 256 CLAUDS objects.

Its historical exploratory status must be retained.

### T2 — New independent CLAUDS spectroscopic confirmation

Use only if a new, properly separated spectroscopic sample is available.

### T3 — New CLAUDS ZPHOT agreement cohort

Optional.

Must be disjoint from the historical 256 diagnostic objects and frozen before accessing its ZPHOT values.

This is not a replacement for T2.

Do not evaluate model performance on T2 or T3 during model development.

## 8. Freeze the benchmark definitions

Before Stage 3B training or independent target evaluation, create a machine-readable benchmark specification.

It must define:

- Dataset versions and hashes.
- Catalogue inclusion criteria.
- Source partitions.
- Identity grouping.
- Spectroscopic-quality cuts.
- Evaluation cohorts.
- Redshift range and binning.
- Missing-band policy.
- Flux and uncertainty conventions.
- Source-only preprocessing.
- Primary statistical comparisons.
- Secondary analyses.
- Bootstrap units.
- Random seeds.
- Model families.
- Allowed source-only model-selection rules.
- Explicit failure and incompleteness criteria.

Use the previous 300-bin redshift grid spanning 0–4.5 unless there is a source-only justified reason for changing it.

Objects outside the supported range must not silently be clipped into the final bin.

Any changed grid must be separately labelled.

### Primary endpoint

For a genuinely independent spectroscopic target cohort, the primary comparison should be the paired difference in catastrophic-outlier fraction:

`N1_kernel - N1`

using:

`abs(z_pred - z_spec)/(1+z_spec) > 0.15`

Negative differences favour N1_kernel.

Predefine the point estimator derived from each redshift PDF.

### Secondary endpoints

Include:

- NMAD.
- Redshift bias.
- CE and CRPS.
- Nominal 68% coverage.
- PIT/calibration.
- Prediction concentration.
- Model failure rates.
- Redshift-conditioned performance.
- Brightness/S/N-conditioned performance.
- Filter-availability-conditioned performance.

Define success and interpretation rules before opening new target labels.

Use paired bootstrap at galaxy identity level.

Report uncertainty across independent model-training seeds separately.

Do not pool repeated views as independent galaxies.

If the spectroscopic confirmation cohort is too small for useful subgroup inference, report that limitation rather than inferring a robust high-z result.

## 9. Stage 3A execution gate

At the end of Stage 3A, categorise readiness.

**Ready for full Stage 3B:**

- An independently verified Legacy training population is available.
- Source/target identity and split audits pass.
- Photometric interfaces and response metadata are verified.
- Source coverage is quantified.
- A frozen evaluation protocol exists.
- Independent spectroscopic target confirmation is available, or the absence of such confirmation is explicitly accepted as a limitation of a non-confirmatory training experiment.

**Ready for limited Stage 3B pilot:**

- Valid source training and validation are available.
- Full mixed-source or independent target confirmation is incomplete.
- Source-only architecture experiments are still scientifically interpretable.

**Blocked:**

- Data leakage cannot be resolved.
- Photometric units or filter identities remain unverified.
- Spectroscopic truth is ambiguously linked to observations.
- Source/validation separation is unreliable.

Do not launch a misleading population-scale experiment when only the paired-source pilot is available.

Produce a readiness report regardless of the gate outcome.

---

# Part II — Stage 3B: Real-Survey Physics-Informed Representation

## 10. Experimental hypothesis

The primary hypothesis is:

**Providing a numerical photo-z model with explicit redshift-dependent SED/filter response information improves transfer to new observational systems beyond ordinary filter-aware numerical representations.**

This hypothesis concerns physics-informed representation.

It does not establish a foundation-model advantage.

The experiment must separately test three possible explanations for an observed gain:

1. A more effective input architecture.
2. Additional external physical information.
3. Increased model capacity or easier optimisation.

Do not combine them into one claim.

## 11. Principal architectures

Use four core model configurations.

### Model R0 — Conventional numerical baseline

A compact fixed-slot numerical MLP.

Use stable wavelength-based band canonicalisation, not the historical random band-to-slot assignment.

Inputs:

- Observed fluxes.
- Uncertainties.
- Availability masks.
- Filter identities or explicitly matched response information.
- Separately controlled brightness information.

This is a secondary baseline.

### Model R1 — Shared band-set encoder

Reuse the validated N1 architecture.

Each band is encoded using:

- Normalised flux.
- Normalised uncertainty.
- Availability and quality.
- Numerical filter-response information.

Use shared per-band weights and permutation-invariant aggregation.

The target redshift output is a normalised PDF.

This is the primary comparison baseline.

### Model R2 — Physics-kernel-informed set encoder

Reuse and validate the Stage 2B.6 N1_kernel architecture.

For filter b, SED basis k and redshift grid point z, use the verified forward response:

`K[b, z, k]`

constructed by integrating the rest-frame SED basis through the actual response curve.

This representation must come from independently verified physical templates and response integration, not target redshift labels.

Use the numerical kernel as an explicit per-band physical representation.

Ensure:

- All bands use the same representation convention.
- New filters can be incorporated through their measured response curves without retraining filter-position-specific parameters.
- Signed measured fluxes remain signed.
- Missing bands remain masked.
- No redshift labels enter the kernel.
- No galaxy-specific target truth enters the SED basis.
- Source-only normalisation is preserved.

The main output remains a supervised redshift PDF.

Do not automatically impose the previous N2 profile-likelihood score on its final logits.

The purpose is to test **physical information as representation**, rather than physical likelihood as a potentially incorrect hard constraint.

### Model R3 — Physical-only baseline

Retain source-fixed physical fitting configurations established in Stage 2A–2A.6.

At minimum include the original D12 reference and a clearly labelled constrained diagnostic where feasible.

Do not describe D1 or D2 as a universally calibrated physical likelihood.

The physical-only model need not have the same trainable parameter budget, but its extra SED prior information must be disclosed.

## 12. Mandatory capacity and information controls

This is crucial.

Previous N1_kernel improvements were not an identical-information or identical-capacity comparison.

Introduce targeted controls.

### C1 — Capacity-matched R1

Increase the hidden width or other nonphysical model capacity of R1 to approximately match R2's trainable parameter count.

Keep the same observations and source supervision.

### C2 — Full-response-information R1

Provide R1 with a sufficiently expressive numerical encoding of complete filter response curves, rather than only short summaries.

Use a parameter budget comparable to R2 where practical.

This tests whether detailed filter-response access, without the external SED basis, explains much of the gain.

### C3 — Alternative kernel basis

Construct one predeclared alternative response-kernel representation using a fixed set of generic smooth spectral basis functions.

Match the approximate number and shape of input kernel channels.

This tests whether the advantage requires astrophysically meaningful SED basis spectra or mostly benefits from a richer wavelength/redshift representation.

Do not claim this control perfectly matches the information content of an FSPS basis.

### C4 — Physical metadata counterfactual

Evaluate correctly matched responses against controlled wrong-response metadata.

Where possible distinguish:

- Wrong metadata with fixed measured fluxes.
- Physically regenerated photometry through alternative response curves.

Only the latter provides a physically consistent counterfactual observation.

Wrong-metadata sensitivity alone is not evidence that the model has learned astrophysical physics.

Keep these controls small and pre-specified.

Do not build a large architecture search.

## 13. Training configurations

Use three scientifically distinct source-training setups.

### Experiment E1 — HSC source to Legacy source

Train on HSC spectroscopic source galaxies.

Evaluate source-side transfer on independent Legacy spectroscopic galaxies.

Separate overlapping paired objects from genuinely independent population evaluation.

### Experiment E2 — Legacy source to HSC source

Train on recovered Legacy spectroscopic source galaxies.

Evaluate transfer on independent HSC spectroscopic galaxies.

Use identical object-overlap exclusion rules.

### Experiment E3 — Mixed source to CLAUDS

Train on deduplicated HSC plus Legacy spectroscopic data.

Evaluate on CLAUDS only after source model selection and benchmark locking.

This is the primary real target transfer experiment.

For E3, provide:

- HSC-only source baseline.
- Legacy-only source baseline.
- Mixed-source model.
- Explicit survey balancing.
- Source redshift distribution.
- Source magnitude/S/N distribution.

Do not assume that more source observations necessarily improve target-domain coverage.

Do not let duplicated galaxy views silently dominate the mixed-source objective.

If independent Legacy training is unavailable, E3 remains incomplete.

Run E1 or a small paired-view source diagnostic where justified, but do not claim completion of E3.

## 14. Input normalisation and brightness

Stage 0.5 showed that the archived Qwen model was strongly sensitive to absolute flux scale.

For R0/R1/R2, use a consistent source-defined numerical input contract.

The primary configuration should separate:

- Relative photometric spectral shape.
- Relative measurement uncertainty.
- Absolute flux scale.
- Valid-band availability.
- Physical filter response.

Use a stable per-object flux normalisation that supports negative measurements.

Do not remove S/N information inadvertently.

Keep absolute brightness as a separately controlled feature.

Pre-register:

- Primary amplitude-normalised model.
- A limited source-only brightness-feature ablation.

Do not use target labels to select whether brightness is helpful.

Source-to-target magnitude extrapolation must be explicitly reported.

A model using brightness as a population prior is not inherently unphysical, but that prior may fail when the survey selection function changes.

## 15. Training and optimisation

Use matched data partitions and supervision counts for the primary R1/R2 comparisons.

Requirements:

- Same source labels.
- Same training objects.
- Same validation objects.
- Same redshift grid.
- Same batch construction.
- Comparable optimisation budget.
- Multiple independent seeds.
- Source-only checkpoint selection.
- No target-label-based tuning.

Start with compact models and a deterministic pilot.

Measure:

- Training CE.
- Validation CE.
- Outliers and NMAD.
- Gradient stability.
- Output concentration.
- Convergence curves.
- Peak memory.
- Runtime.

Do not repeat the Stage 2B synthetic mistake of declaring a model comparison valid when all models select update zero.

Do not declare full convergence merely because validation CE improves from initialization.

Use common-update and source-convergence comparisons.

If one architecture systematically underfits, identify that limitation before making transfer claims.

Do not expand into a large training sweep unless the source-only pilot is valid and the Stage 3A data-readiness gate supports it.

## 16. High-redshift supervision analysis

This is an essential scientific component.

For every model report performance separately at:

- z < 1.
- 1 <= z < 2.
- z >= 2.

Also preserve finer intervals when sample sizes permit.

Record the number of independent source-training galaxies supporting each range.

Separate:

**Interpolation within source redshift support**

from:

**Extrapolation beyond source redshift support.**

Do not claim learned high-z generalisation merely because a model was evaluated on high-z target objects.

Use high-z source spectroscopy where available.

If no adequate source high-z population exists, explicitly classify the high-z experiment as unsupported or exploratory.

Optional synthetic high-z augmentation may be conducted as a separate experiment but cannot replace independent real high-z spectroscopic supervision.

## 17. Source-validation gate

Before any independent CLAUDS evaluation, verify:

1. R1 and R2 both learn the source task.
2. Both have meaningful source-validation performance.
3. Their output PDFs are finite and normalized.
4. Masking and band permutation invariance pass.
5. Response integration and kernel hashes are correct.
6. Source-only checkpoint selection is complete.
7. Model-training seeds are fixed.
8. All comparison controls are frozen.
9. All target evaluation metrics are registered.
10. All model and protocol hashes are written.

If this gate fails, report the source diagnosis without opening new target labels.

Do not choose a target checkpoint after observing CLAUDS performance.

## 18. Real target evaluation

If independent CLAUDS spectroscopy T2 is available, evaluate the frozen candidate models.

This is the primary scientific confirmation.

If only T1/T3 ZPHOT references are available, report an explicitly non-confirmatory photometric-reference agreement experiment.

For each eligible target cohort, report:

- NMAD.
- Catastrophic-outlier fraction.
- Bias.
- CE and CRPS.
- 68% coverage.
- PIT diagnostics.
- Predicted redshift distribution.
- Prediction collapse concentration.
- Failure and insufficient-band rates.
- Conditional metrics by brightness, S/N, redshift and band availability.

Compare the same target objects across all models.

Use paired bootstrap intervals and separate training-seed uncertainty.

For the primary R2–R1 comparison, report:

- Absolute outlier difference.
- Relative outlier change.
- Paired confidence interval.
- NMAD difference.
- CRPS and CE differences.
- Whether improvements persist across seeds.
- Whether improvements are limited to particular redshift/S/N regimes.

Do not report significance from treating different model predictions on the same galaxy as independent observations.

### Full CLAUDS versus restricted-band inference

Compare:

- Full available CLAUDS UV/optical/NIR measurements.
- HSC-compatible grizy.
- A predeclared subset with wavelength overlap to source systems.

Use actual matching responses.

Do not silently substitute filters with similar names.

Distinguish the benefit of new wavelength coverage from increased band count and changes in photometric quality.

Report results conditional on usable-band count.

### Target-selection effects

An independent spectroscopic CLAUDS subset is likely to have a nontrivial selection function.

Audit how its magnitude, redshift, S/N and colour distributions compare with the full unlabeled CLAUDS photometric population.

Do not describe performance on a bright or spectroscopically selected subset as representative of all faint CLAUDS galaxies without supporting evidence.

## 19. Archived Qwen and previous baselines

The original scientific project concerns foundation-model posttraining.

Preserve its relevance without prematurely launching Stage 2C.

Where technically feasible, run the existing frozen HSC Qwen checkpoint on exactly the same eligible target objects under its documented original interface.

Do not retrain or tune it.

If the original Qwen interface cannot validly process a target observational configuration, report the limitation.

Do not compare an HSC-source Qwen score on one target cohort against R2 evaluated on a different cohort as if it were a paired benchmark.

Also retain the source-fixed EAZY/physical baseline with documented posterior conventions.

Do not infer foundation-model pretraining benefits from R2 performance.

R2 is a numerical physics-informed architecture, not a pretrained language foundation model.

## 20. Scientific decision criteria

The final assessment must separately address five claims.

### Claim A — Numerical representation

Does a mask-aware band-set encoder outperform a conventional fixed-slot MLP on real heterogeneous photometry?

### Claim B — External physics information

Does an explicit SED-response kernel improve performance over R1 and capacity-matched/full-response controls?

### Claim C — Real cross-survey transfer

Does the improvement persist on independent real survey observations, especially CLAUDS?

### Claim D — High-redshift generalisation

Is performance supported by real source spectroscopy at comparable redshifts, or is the model extrapolating beyond the labelled population?

### Claim E — Foundation-model value

Does any result demonstrate that pretrained Qwen contributes additional knowledge beyond the physical and numerical controls?

Stage 3 cannot establish Claim E without a separate matched foundation-model experiment.

Do not claim it has been established.

## 21. Failure interpretation

If R2 fails, determine whether the likely limitations include:

- Inadequate source redshift coverage.
- Photometric domain shift.
- Mismatched SED basis information.
- Different measurement conventions.
- Weak cross-filter invariance.
- Training instability.
- Population selection effects.
- Intrinsic redshift degeneracy.

Use controlled evidence.

Do not assign causal percentages without experiments that isolate the mechanisms.

If R2 improves only source-domain NMAD but not target transfer, do not call it a cross-survey generalisation success.

If R2 improves point estimates but catastrophically worsens PDF calibration, report a mixed result.

If it works on a bright spectroscopic CLAUDS subset but not the faint population, report the selection limitation.

If R2 only beats R1 because it receives external SED information, describe the result as a benefit of physics-informed input information, not automatically as an architectural improvement.

## 22. Execution and resource discipline

Run Stage 3A first.

Do not initiate a definitive Stage 3B population experiment if the required data are unavailable or contaminated.

A limited source-only Stage 3B pilot is acceptable when it addresses a scientifically identifiable question.

Prefer:

1. Data recovery and audits.
2. Small source-only model pilot.
3. Frozen architecture and configuration selection.
4. Independent source cross-survey evaluation.
5. Independent CLAUDS confirmation, where available.

Do not spend large GPU resources before the source data and evaluation design are verified.

Do not interrupt existing jobs.

No Qwen training, Stage 2C or large adapter sweep.

Do not overwrite Stage 0–2 results.

## 23. Reproducibility and deliverables

Create separate directories, for example:

`filter-aware-runs/1029/stage3a-real-survey-readiness/`

`filter-aware-runs/1029/stage3b-kernel-transfer/`

### Stage 3A deliverables

1. Recovered Legacy dataset or documented recovery blocker.
2. Verified source catalogue manifests.
3. HSC/Legacy/CLAUDS overlap audit.
4. Spectroscopic source-redshift coverage.
5. Magnitude/S/N/colour support diagnostics.
6. Available real target spectroscopic matches.
7. Independent confirmation-cohort definitions.
8. Fixed benchmark protocol and its hash.
9. Training/validation/test membership manifests.
10. Explicit readiness verdict.

### Stage 3B deliverables

1. R0/R1/R2/R3 architecture definitions.
2. Capacity and information-access controls.
3. Response-kernel and template provenance.
4. Source training histories and convergence diagnostics.
5. Source cross-survey results.
6. Mixed-source results where eligible.
7. CLAUDS confirmation or explicitly labelled exploratory results.
8. Redshift-, brightness- and S/N-conditioned metrics.
9. Paired bootstrap and seed variability.
10. Runtime and GPU-memory comparisons.
11. Exact model checkpoints, configuration hashes and reproduction commands.
12. Scientific interpretation with negative and inconclusive results preserved.

### Final joint scientific assessment

Answer:

1. Was the independent Legacy training population recovered?
2. What is the real high-z spectroscopic coverage of the usable source datasets?
3. Is independent CLAUDS spectroscopic confirmation available?
4. Does N1_kernel improve real source cross-survey transfer over the conventional N1 model?
5. Does that improvement survive capacity and information-access controls?
6. Does it transfer to independent CLAUDS galaxies, if such evaluation is possible?
7. Is the benefit present in the high-z and faint regimes relevant to CLAUDS?
8. Does the experiment identify a credible physical representation advantage?
9. What remains necessary for a scientifically defensible foundation-model comparison?
10. Should Stage 2C with pretrained Qwen be launched, redesigned or deferred?

**Execute all feasible Stage 3A work and the scientifically justified Stage 3B experiments. Do not stop after writing a plan. Do not invent missing data, claim unavailable spectroscopy, or turn previously inspected CLAUDS diagnostics into independent confirmation.**

**Proceed autonomously from Stage 3A to Stage 3B only if the specified correctness and data-readiness gates permit a valid experiment.**