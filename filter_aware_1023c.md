# Experiment C — Warm-start Level-2 Filter-Physics-Aware QLoRA

Implement and execute a continued-SFT experiment using Level-2 photometric input representation.

Start from the **same original archived GalaxiesML-Spec HSC Qwen3.5-4B-Base QLoRA checkpoint and trained PDF head used as the initialisation for Experiments A and B**.

Do not continue from the trained Level-1 checkpoint. Experiments A, B and C must be three independent branches from an identical starting model.

## Scientific objective

Investigate whether supplying basic physical information about photometric filters improves the ability of an already post-trained foundation model to interpret observations from different photometric systems.

Level 2 differs from Level 1 by including:

1. Filter identity;
2. Pivot wavelength;
3. Rectangular equivalent bandwidth.

The critical comparison is **Level 2 versus Level 1**, because Level 1 already provides explicit band labels.

Do not assume that filter metadata will improve performance.

## Metadata

Use the previously implemented and tested filter-response registry:

`configs/filter_registry_v1.json`

and its associated response profiles.

Verify the definitions, wavelength units and filter-to-survey mapping.

Use the actual relevant HSC filters for source training and the correct filter identities for Legacy and CLAUDS evaluation.

Preserve documented approximations, including existing limitations involving HSC r/r2 and i/i2 exposure mixtures.

Do not fabricate filter properties.

Do not introduce full transmission curves, new learned filter embeddings or additional physical descriptors in this experiment.

## Data and checkpoint

- Qwen3.5-4B-Base.
- Original completed HSC rank-8 QLoRA adapter.
- Original trained 256-hidden-unit PDF head.
- 22,444 HSC training objects.
- 5,683 HSC validation objects.
- Original 300-bin redshift grid over 0–4.5.
- Identical object splits and supervision labels to Experiments A and B.

Both adapter and head must be warm-started from the archived checkpoint.

No random prediction-head initialisation.

## Serialization

Use the existing Level-2 serializer.

Example:

```text
Survey: HSC

Filter: HSC_g
Pivot wavelength: ... nm
Rectangular equivalent width: ... nm
Flux: ... nanomaggies
Uncertainty: ... nanomaggies

Filter: HSC_r
Pivot wavelength: ... nm
Rectangular equivalent width: ... nm
Flux: ... nanomaggies
Uncertainty: ... nanomaggies

...

EBV: ...
```

The precise format should match the previously implemented Level-2 configuration.

**Do not change the formatting, flux representation or error convention relative to Level 1 except for the additional wavelength and bandwidth metadata.**

Otherwise the Level-1 versus Level-2 comparison will not isolate the contribution from filter physics.

Check actual tokenizer inputs, token counts, maximum sequence length and truncation.

Increase the common maximum sequence-length setting across experimental arms if required to prevent truncation, while keeping other settings matched.

## Before training

1. Restore the original HSC QLoRA checkpoint and PDF head.
2. Run source validation immediately with Level-2 inputs before any updates.
3. Measure the performance penalty from introducing the new physical metadata.
4. Verify filter metadata correspondence for individual observations.
5. Confirm that the complete source cohort and split fingerprint match Experiments A and B.

## Continued SFT

Use the same protocol:

- Maximum 2 additional epochs.
- No head-only warmup.
- Adapter LR = 3e-6.
- Head LR = 1e-5.
- Effective batch size = 16.
- Seed = 1023.
- Original rank and adapter target modules.
- Validation approximately every 250 updates.
- Same optimisation and checkpoint-selection procedure as A/B.
- Select the best checkpoint by source-validation CE, including the zero-update baseline.

Do not change the architecture, PEFT module targets, trainable parameter budget or redshift grid.

## Evaluation

Evaluate the selected model on:

1. HSC source validation.
2. DESI–Legacy target validation.
3. CLAUDS target validation.

Reuse exactly the same target query objects as Experiments A and B.

No target-label fitting, recalibration or test-set-based optimisation.

Report CE, NMAD, outlier fraction, bias, CRPS, 68% coverage, training steps and computational cost.

Compare against both:

- Original archived HSC QLoRA;
- Matched Level-0 and Level-1 continued-SFT runs.

Distinguish source-domain recovery from genuine cross-survey improvement.

A model that fails to recover its original source-domain predictive quality cannot provide a clean test of whether filter metadata improve transfer.

## Interpretation

Since every HSC source-training object uses essentially the same fixed filter set, wavelength and bandwidth do not vary independently across source galaxies.

Consequently, successful HSC-only adaptation does not demonstrate that the model has learned a physically transferable dependence on filter response.

Even if Level 2 improves target metrics, treat this as preliminary evidence requiring controls involving multiple training filter systems and, ideally, shuffled or incorrect metadata.

The present experiment is an initial controlled evaluation, not a demonstration of universal filter-aware generalisation.

## Output

Save independently under:

`filter-aware-runs/1023/hsc-warmstart/level2/`

Preserve prior runs and checkpoints.

Do not interrupt other running training jobs.

Implement and execute the experiment, not merely describe it.

The final report must answer:

**Does adding actual filter wavelength and bandwidth information provide measurable transfer benefits beyond explicit band identification, after both representations have been adapted from the same existing high-performance QLoRA checkpoint?**