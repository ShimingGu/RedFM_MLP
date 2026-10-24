# Implement Level-2 Filter-Aware Photometric Input for Cross-Survey Photo-z Transfer

## Scientific objective

We are studying whether post-trained general foundation models can perform photometric-redshift inference across heterogeneous astronomical surveys.

Our existing Qwen3.5-4B-Base models achieve strong in-domain performance but perform poorly under raw cross-survey zero-shot transfer.

The current observational systems include:

- **CLAUDS:** Existing AB-magnitude-based input pipeline. Inspect the actual loader to determine the exact bands, conventions, and missing-value handling.
- **GalaxiesML-Spec:** HSC grizy, using `cmodel_mag` and `cmodel_magsigma`, converted to flux and flux uncertainty, plus EBV. Labels are DESI spectroscopic redshifts.
- **DESI–Legacy:** Legacy DR9 grzW1W2, using Tractor flux and inverse variance, plus EBV. Labels are DESI DR1 Iron spectroscopic redshifts.

The existing source-trained QLoRA models exhibit severe raw cross-survey degradation, sometimes collapsing to nearly constant redshift predictions.

However, current comparisons mix differences in filter systems, numerical input conventions, observational selection and target labels.

**Your task is to implement a physically meaningful Level-2 filter-aware input representation and experimentally determine whether it improves cross-survey transferability.**

The scientific question is:

**Does explicitly providing the identity and basic physical characteristics of each photometric filter help a post-trained foundation model generalise across different observational systems?**

Do not assume the answer is yes.

---

## 1. Inspect the existing pipeline

Before changing anything, identify:

1. The Qwen3.5-4B checkpoint actually used by the existing completed runs.
2. Existing CLAUDS, GalaxiesML-Spec and DESI–Legacy data loaders.
3. The exact current photometry serialization format.
4. The QLoRA and DoRA training implementations.
5. Prediction-head architecture, redshift grid, losses and evaluation metrics.
6. Existing training/validation/test splits and object identifiers.
7. Current context-length limit, tokenizer and truncation statistics.

Reuse existing functionality wherever possible.

Do not modify previously completed experiments, checkpoints, configurations or results.

Do not interrupt running GPU jobs.

## 2. Obtain real filter metadata

For every photometric band actually used by the three catalogue loaders, identify:

- Survey/instrument;
- Exact filter identifier;
- Effective or pivot wavelength;
- Effective bandwidth;
- Units;
- Source of the filter transmission information.

Prefer authoritative measured filter transmission curves, such as official survey documentation or the SVO Filter Profile Service.

**Do not invent approximate wavelengths or bandwidths and present them as physical metadata.**

Choose explicit, reproducible definitions for wavelength and bandwidth. Prefer response-based quantities independent of the observed object's SED, such as pivot wavelength and a consistently defined effective width.

Explain the definitions and use them consistently.

Important considerations:

- HSC g and Legacy g are not necessarily identical filters.
- Legacy optical bandpasses may depend on the imaging instrument or footprint.
- WISE W1/W2 must use their actual infrared filter properties.
- Confirm the filter system used by CLAUDS rather than assuming that all bands with identical names share transmission curves.
- Distinguish photon-counting response conventions when necessary.

If exact instrument/filter identification is unavailable for some objects, use a documented, survey-level approximation only if scientifically defensible. Explicitly record the approximation.

If reliable metadata cannot be established, report the limitation instead of inventing values.

Store the filter metadata in a reusable, version-controlled configuration file with provenance.

## 3. Implement three input representations

Implement a configurable serializer supporting:

### Level 0 — Existing representation

Preserve the original input format exactly.

This provides backward compatibility with existing runs.

### Level 1 — Explicit band identity

For example:

```text
Survey: HSC
Photometry:
HSC_g: flux=..., sigma=...
HSC_r: flux=..., sigma=...
HSC_i: flux=..., sigma=...
HSC_z: flux=..., sigma=...
HSC_y: flux=..., sigma=...
EBV: ...
```

The values above are illustrative. Preserve the actual catalogue measurements and units.

Every measurement must be attached to its correct physical band, not merely its feature position.

### Level 2 — Band identity + physical filter metadata

For example:

```text
Survey: HSC

Photometry:
Filter: HSC_g
Pivot wavelength: ... nm
Effective bandwidth: ... nm
Flux: ... nanomaggies
Flux uncertainty: ... nanomaggies

Filter: HSC_r
Pivot wavelength: ... nm
Effective bandwidth: ... nm
Flux: ... nanomaggies
Flux uncertainty: ... nanomaggies

...

EBV: ...

Predict the redshift probability distribution.
```

The exact wording can be optimised for token efficiency, but all three physical components must be explicit:

1. Filter identity;
2. Filter wavelength;
3. Filter bandwidth.

Keep a stable field format and numeric precision across surveys.

Do not accidentally present W1/W2 as HSC i/z/y or relabel features based on array positions.

The primary experimental distinction between Level 1 and Level 2 must be the additional physical filter metadata, not unrelated changes in target definitions or preprocessing.

## 4. Handle photometric numerical conventions correctly

Inspect the existing loaders and document how each catalogue represents:

- Fluxes or magnitudes;
- Measurement uncertainties;
- Negative and low-S/N fluxes;
- Missing measurements;
- Galactic extinction information.

Provide an optional common flux-based representation for controlled experiments.

Do not silently discard valid negative fluxes.

Do not use logarithmic magnitude conversion on non-positive fluxes.

If magnitude-to-flux transformations are used, propagate uncertainties consistently and clearly describe any approximation.

Keep the original numerical conventions available so that the effect of numerical harmonisation can be separated from the effect of filter metadata.

In particular, adding wavelength metadata while simultaneously changing CLAUDS magnitudes into fluxes introduces two experimental changes. These must not be attributed jointly to filter metadata.

## 5. Make the representation usable by the existing Qwen training pipeline

Integrate all three levels with existing QLoRA/DoRA training and evaluation.

Requirements:

- Use the same Qwen checkpoint as the existing experiments.
- Preserve the existing redshift-PDF prediction head.
- Preserve the original target redshift grids where necessary for checkpoint replay.
- Permit a configurable `max_seq_length`.
- Measure actual token lengths after serialization.
- Verify that complete input records are not inadvertently truncated.
- Preserve object identifiers, missing-value masks and feature validity.
- Use the exact same data partitions for controlled comparisons.
- Record the serializer version and filter metadata version in every run configuration.

Do not increase model parameter count merely to accommodate the new metadata.

Do not change adapter targets, ranks, learning rates or unrelated optimisation settings as part of the initial controlled test.

## 6. Validate the implementation before expensive training

Implement automated tests checking:

1. Correct mapping of every flux/error pair to its physical filter.
2. Correct wavelength and bandwidth lookup.
3. Consistent wavelength units.
4. Correct flux/error units.
5. Correct missing-value handling.
6. Preservation of source redshift labels.
7. Absence of spectroscopic redshift leakage into model inputs.
8. Identical object membership and split identity between experimental conditions.
9. Token-length and truncation behaviour.
10. Backward compatibility of Level 0 with archived source predictions.

Print representative final tokenizer inputs from each catalogue, including at least one record with missing or low-S/N measurements.

Verify the real text sent to the tokenizer, not merely the intermediate data arrays.

## 7. Experimental plan

After the implementation passes validation, run a small, controlled pilot.

Avoid launching a large training matrix before the pilot has established that the implementation is correct.

### Experiment A — Input-representation ablation

On identical training cohorts, compare:

- Level 0: existing serialization;
- Level 1: explicit filter identities;
- Level 2: filter identities + wavelength + bandwidth.

Start with Qwen3.5-4B-Base QLoRA.

Use matched splits, training budgets, adapter configuration and evaluation procedures.

Compare:

- In-domain validation CE;
- NMAD;
- Catastrophic-outlier fraction;
- Bias;
- CRPS;
- 68% coverage;
- Predicted redshift distributions.

Use validation data for checkpoint selection. Keep held-out test data untouched until selecting the final experimental configurations.

### Experiment B — Cross-survey evaluation

Evaluate source-trained models on the other observational systems.

At minimum consider:

- GalaxiesML-Spec → DESI–Legacy;
- DESI–Legacy → GalaxiesML-Spec;
- CLAUDS → GalaxiesML-Spec;
- GalaxiesML-Spec → CLAUDS.

Extend the matrix where data and compatible configurations permit.

Clearly distinguish:

- Raw-interface transfer;
- Numerically harmonised transfer;
- Level-1 band-labelled transfer;
- Level-2 filter-aware transfer.

Keep weights fixed during zero-shot target evaluation.

Do not perform target calibration or target-label fitting.

Use the same target objects when comparing different serializers.

Perform appropriate source/target object-overlap exclusions for meaningful transfer results.

CLAUDS uses photometric redshift reference labels in the existing experiments, whereas the other two catalogues use DESI spectroscopic labels. Report this distinction explicitly.

### Experiment C — Mixed-survey training (optional follow-up)

If Levels 1 and 2 are functional but source-only transfer remains poor, prepare a controlled mixed-survey QLoRA pilot.

Use observations from at least two photometric systems with correct filter metadata and catalogue identity.

Evaluate separately on each constituent survey and, where feasible, on a held-out observational system.

Keep separate results for:

1. Source-only zero-shot transfer;
2. Supervised mixed-survey training;
3. Transfer to an unseen survey.

Do not conflate these three capabilities.

## 8. Critical scientific interpretation

There is an important identifiability limitation.

Within a single survey, each band's wavelength and bandwidth are constant across all training galaxies.

Therefore, successful single-survey training does not demonstrate that the model has learned a physically meaningful relationship between filter characteristics and redshift.

Similarly, success on two survey systems jointly used for training does not establish generalisation to unseen filters.

The experiments should distinguish:

- Memorising the source survey's input conventions;
- Learning useful band identities;
- Learning relationships involving physical filter metadata;
- Adapting across measurement systems;
- Genuine transfer to unseen observational configurations.

Level 2 may fail to improve transfer. This is an acceptable scientific outcome.

We specifically want to determine whether the additional metadata is useful, rather than merely making the prompt longer.

## 9. Deliverables

Produce:

1. A concise description of the existing pipeline and necessary modifications.
2. The implemented configurable serializers.
3. A verified filter-metadata registry with sources and definitions.
4. Automated serializer/data-consistency tests.
5. Representative actual tokenizer inputs for all catalogues.
6. Token-length and memory-cost comparisons.
7. A small Level-0/1/2 experimental comparison if computational resources permit.
8. A cross-survey evaluation table for completed pilot configurations.
9. A short scientific interpretation separating observed results from hypotheses.

Save code, configurations, validation outputs and reports within the existing project structure, without overwriting earlier experiments.

Prioritise implementation, validation and a modest pilot. Do not launch expensive full-scale runs while existing training jobs require the GPU.

**Work autonomously through repository inspection, implementation, unit tests and low-cost validation. Do not stop after producing a proposal. If a substantial GPU experiment cannot be run safely, leave a reproducible launch command and clearly identify the unexecuted experiment.**

The final report must make clear what was actually implemented, tested and measured.
