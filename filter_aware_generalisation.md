# Codex Task — CLAUDS Common-Flux Interface Harmonisation and Fixed-Weight Transfer

## Objective

Investigate whether the catastrophic failure of HSC-trained Qwen3.5-4B-Base models on CLAUDS can be mitigated by harmonising the numerical input representation.

**CLAUDS is the primary scientific target. Do not optimise for HSC ↔ DESI-Legacy transfer.**

We have three completed HSC-source warm-start QLoRA experiments:

- Experiment A: Level 0, original input representation.
- Experiment B: Level 1, explicit band identities.
- Experiment C: Level 2, band identities plus pivot wavelength and rectangular equivalent bandwidth.

All three checkpoints are already trained. Their source performance and native-input CLAUDS transfer results are documented in `report_fwa.md`, `report_fwb.md`, and `report_fwc.md`.

Currently, the selected A/B/C models have approximately 96–97% catastrophic outliers when evaluated on CLAUDS, despite good HSC source-domain performance.

CLAUDS uses native AB magnitudes, while the HSC-source models were trained on flux-based inputs.

An optional CLAUDS common-flux representation was implemented in the previous filter-aware pilot but has not yet been evaluated with the completed A/B/C checkpoints.

**Execute this evaluation now, using existing trained weights.**

## 1. Hard constraints

- No additional SFT, QLoRA, DoRA, RLVR or any other training.
- Do not modify adapter weights, PDF heads or base models.
- Do not fit any calibration using CLAUDS labels.
- Do not select model checkpoints or tune preprocessing using target-domain performance.
- Do not overwrite existing models or experiment results.
- Do not evaluate the held-out test sets.
- Reuse the existing 256 audited CLAUDS validation objects, with identical ordering and identifiers.
- Retain the HSC-source PDF grid of 300 bins over redshift 0–4.5.
- Preserve the original source/target overlap exclusions.

The objective is to isolate input-interface effects, not maximise validation performance through experimentation.

## 2. Inspect and validate the CLAUDS common-flux representation

Start by inspecting:

- `aion_magnitude/filter_photometry.py`
- `configs/filter_registry_v1.json`
- `notebooks/qwen_filter_aware.py`
- The existing CLAUDS loader, native serializer and optional common-flux implementation.
- The completed Experiment A/B/C checkpoints and manifests.

Verify that the flux transformation uses:

\[
f_{\rm nMgy}=10^{(22.5-m_{\rm AB})/2.5}.
\]

For each CLAUDS measurement, ensure:

1. Magnitudes are genuinely on the AB system before conversion. Check the original catalogue conventions, especially near-infrared bands.
2. Fluxes are expressed in nanomaggies, consistent with the HSC source representation.
3. Actual filter identifiers are retained, including survey/instrument distinctions.
4. Missing values remain explicitly marked as missing.
5. Existing imputed magnitudes are not silently treated as genuine observations. Preserve original missingness flags where available.
6. Missing photometric uncertainties are not replaced by zero or fabricated values.
7. If magnitude uncertainties exist, propagate them using the appropriate Jacobian.
8. If an uncertainty is unavailable, use the established explicit missing-value convention.
9. Do not invent EBV values or extinction corrections.
10. The transformation does not introduce redshift information into the input.

Check whether CLAUDS magnitudes have already been corrected for Galactic extinction. Document any mismatch with the HSC training convention rather than silently changing the data.

If the existing common-flux implementation violates any of these conditions, make the minimum necessary correction and add regression tests.

Preserve the original native-magnitude serializer unchanged.

## 3. Evaluate existing checkpoints

Use the selected checkpoints from Experiments B and C.

Experiment A provides the unchanged-input control; reuse its archived CLAUDS result and verify replay where necessary.

Run the following fixed-weight evaluation matrix:

| Model | CLAUDS input | Purpose |
|---|---|---|
| A — Level 0 | Original native input | Archived control |
| B — Level 1 | Native AB magnitudes | Existing reference |
| B — Level 1 | Harmonised nanomaggy flux | Numerical-interface test |
| C — Level 2 | Native AB magnitudes | Existing reference |
| C — Level 2 | Harmonised nanomaggy flux | Numerical interface + filter physics |

Use exactly the same 256 CLAUDS objects and reference ZPHOT labels.

For B and C, preserve the respective serializer structures and filter metadata. The input representation should change only as required by replacing native photometric magnitudes with physical flux measurements and corresponding validity/error conventions.

Do not inadvertently change band order, band identity, redshift grid or prediction-head weights.

Report explicitly which serialization fields change beyond numerical values (for example `magnitude` becoming `flux`) so the comparison is not incorrectly described as a pure numerical transformation.

## 4. Common-band versus full-band evaluation

Where the actual CLAUDS catalogue permits, repeat B/C common-flux inference under two predetermined band configurations:

**Configuration I — HSC-compatible bands**

Use the CLAUDS bands corresponding to the HSC source's grizy filters, with their actual physical identifiers.

Do not substitute data from another band merely to fill a fixed positional slot.

**Configuration II — Full available CLAUDS photometry**

Use every valid band supported by the CLAUDS input serializer, including available ultraviolet and near-infrared measurements.

Provide correct per-band identifiers and Level-2 metadata.

Both configurations must use identical target galaxies and the same trained checkpoint.

This comparison is intended to distinguish:

- Failure caused by unfamiliar band sets;
- Failure caused by numerical representation;
- Sensitivity to the presence of additional observational measurements.

If Configuration I cannot be constructed faithfully from the existing data, explicitly report why and omit it rather than inventing a mapping.

## 5. Validate tokenizer inputs

Before inference, save representative fully serialized examples for:

- Native CLAUDS Level 1;
- Common-flux CLAUDS Level 1;
- Native CLAUDS Level 2;
- Common-flux CLAUDS Level 2.

Include examples with missing or imputed measurements where available.

Audit:

- Correct magnitude-to-flux conversion;
- Correct band identities;
- Validity masks;
- Units and uncertainties;
- Token counts;
- Maximum sequence length;
- Truncation frequency;
- Exact correspondence between input objects and labels.

The present sequence limit is 512 tokens.

If common-flux input exceeds this limit, do not silently truncate observations. Report the problem and make the smallest safe configuration adjustment, using a consistent rule for the compared conditions.

## 6. Evaluation metrics

For every condition, report:

- Cross-entropy;
- NMAD;
- Catastrophic-outlier fraction;
- Mean and median bias;
- CRPS;
- Nominal 68% interval coverage;
- Predicted median-redshift distribution.

**Do not rely on NMAD alone.**

The existing failure involves near-constant predictions, so explicitly report:

- Median of predicted redshifts;
- 5th–95th percentile range of predicted redshifts;
- Fraction of predictions concentrated near a single redshift;
- Fraction of objects inside the correctly supported redshift range;
- Representative predicted PDFs;
- Predicted versus reference redshift scatter plots.

Use the same outlier definition and PDF conventions as the original experiments.

CLAUDS reference labels are ZPHOT, not spectroscopic redshifts. Retain that distinction throughout the report.

Compute paired differences using identical objects. Query-bootstrap intervals may be included as descriptive uncertainty, but must not be presented as independent model validation.

## 7. Interpret the experiment

Answer these questions in order.

### Question A — Numerical compatibility

Does converting CLAUDS photometry to source-compatible flux units substantially reduce catastrophic prediction collapse?

This is the primary question.

### Question B — Physical band identity

Under common-flux inputs, does Level 1 behave differently from the original Level 0 reference?

Does correct band labelling provide a measurable benefit?

### Question C — Filter metadata

Under common-flux inputs, how does Level 2 compare with Level 1?

Does the apparent benefit of pivot wavelength and bandwidth persist after numerical-interface harmonisation?

### Question D — Band-set mismatch

Does restricting CLAUDS to source-compatible HSC bands behave differently from supplying all valid CLAUDS bands?

Does additional photometry help or destabilise the model?

### Question E — Remaining failure

If predictions still collapse, identify which tested interface changes failed to resolve the problem.

Distinguish observed evidence from possible explanations involving:

- Unseen bands;
- Missing uncertainty information;
- Extinction treatment;
- Source/target photometric distribution shift;
- ZPHOT versus spectroscopic-redshift supervision;
- Source-only adaptation limitations.

Do not attribute persistent failure to intrinsic model capacity without further experiments.

## 8. Execution and outputs

This is a fixed-weight inference experiment. Do not launch new training.

Reuse the existing GPU inference implementation and execute the evaluation when resources are available without interfering with other jobs.

Save results under a new directory such as:

`filter-aware-runs/1023/clauds-common-flux-evaluation/`

Deliver:

1. Input-conversion audit and any necessary minimal fixes.
2. Passing tests for the CLAUDS flux serializer.
3. Actual tokenizer inputs and truncation statistics.
4. A single comparison table for all completed conditions.
5. Paired prediction plots and PDF diagnostics.
6. A concise interpretation stating whether numerical harmonisation rescues CLAUDS transfer.
7. A reproducible manifest of checkpoint identities, input variants, target IDs and evaluation commands.

**The experiment is diagnostic. Do not choose a winning input configuration through target-label tuning, and do not claim unseen-survey generalisation solely from an improvement on these 256 validation queries.**

The main deliverable is a direct answer to:

**Can an already post-trained HSC Qwen model infer meaningful redshifts for CLAUDS when the target photometry is presented in a source-compatible numerical format with correct observational metadata?**