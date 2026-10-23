# Research Task: In-Context Cross-Survey Calibration for Photometric Redshift Inference

## 1. Scientific objective

We are investigating the post-training and cross-survey generalisation capabilities of foundation models for astronomical inference, using photometric redshift estimation (photo-z) as our first case study.

We already have experiments involving Qwen3.5-4B, AION, TabICL, MLPs, and several post-training strategies, including QLoRA, DoRA, IA3, and RLVR.

Our current Paper I focuses on two questions:

1. How do domain-specialised, general-purpose, and tabular foundation models differ in their ability to learn photo-z inference through post-training?
2. Can a model post-trained on one photometric survey adapt to another survey with a different photometric system?

**The second question is the primary focus of this task.**

We have a new idea:

> When applying an already post-trained model to a new survey, provide it with a relatively small spectroscopic calibration sample from that survey. The model should use this calibration sample to adjust its predictions without any additional gradient-based training or parameter updates.

This is closer to **in-context learning, prompt-conditioned adaptation, or inference-time calibration** than conventional fine-tuning.

Our central hypothesis is that a foundation model may be able to exploit a small number of target-survey examples to compensate for differences in photometric systems, even when its parameters remain fixed.

We want to determine whether this works, why it works or fails, and whether it offers a meaningful scientific advantage over conventional approaches.

Do not assume the hypothesis is correct.

## 2. Existing experimental context

We currently have two new spectroscopic training datasets.

### DESI–Legacy

- Spectroscopic labels: DESI DR1 Iron, column `Z`.
- Photometry: Legacy Surveys DR9 Tractor catalogue.
- Inputs: `FLUX_G`, `FLUX_R`, `FLUX_Z`, `FLUX_W1`, `FLUX_W2`, their respective uncertainties derived from `FLUX_IVAR_*`, and `EBV`.
- Total: approximately 300,000 selected galaxies, with roughly 60,000 training, 15,000 validation, and 225,000 test objects.
- Missing photometric bands are explicitly represented.
- Negative fluxes are retained when otherwise valid.

Relevant configuration:

`/arc/home/gsm/aion_output/figures/qwen-desi-ls-qlora-specz-e10/prepared.json`

The original DESI spectroscopic catalogue is located at:

`/arc/projects/ots/Cosmic_Imprint_of_Time/DESI-LS/catalogues/desi_dr1/iron/zcatalog/v1/zall-pix-iron.fits`

### GalaxiesML-Spec

- Spectroscopic labels: `DESI_fibermap/DESI_redshift`.
- Photometry: HSC grizy, using `HSC_metadata/{g,r,i,z,y}_cmodel_mag` and the corresponding `cmodel_magsigma`.
- Magnitudes and uncertainties are converted into flux and flux uncertainties.
- `DESI_fibermap/EBV` is also provided.
- Total: 11 numerical input features.
- This experiment does not use the available Legacy photometry, image pixels, or spectra.

Relevant configuration:

`/arc/home/gsm/aion_output/figures/qwen-galaxiesml-spec-qlora-specz-e10/prepared.json`

**Important:** The two datasets have the same number of input features but represent different photometric systems:

- DESI–Legacy: grzW1W2
- GalaxiesML-Spec: grizy

Their feature positions must not be treated as physically equivalent.

### Current models

QLoRA and DoRA experiments are being run on both datasets, giving a 2×2 comparison with Qwen3.5-4B.

Inspect the actual checkpoints and configurations to determine whether the model is Base or Instruct. Do not assume that all four runs have finished, and do not disturb any running experiments.

Previous reports indicate that Qwen responds strongly to QLoRA/DoRA, whereas AION shows relatively small additional improvements from encoder adaptation beyond a trained attentive head. These are observations from specific experiments, not conclusions about all astrophysical foundation models.

## 3. Your first task: critically assess feasibility

Before implementing the new experiment, inspect the existing repository, model architecture, data pipeline, tokenisation, forward passes, prediction head, and evaluation procedures.

Determine:

**Can our existing post-trained Qwen actually perform in-context adaptation using labelled calibration examples?**

This is not a trivial question.

Our current photo-z implementation may use hidden-state pooling and a learned probabilistic prediction head rather than autoregressive text generation.

Simply prepending calibration examples to a prompt does not necessarily make that head perform in-context learning.

Investigate at least the following:

- How the current numerical features are serialised.
- Whether explicit band identities and survey identities are present.
- How the input sequence is processed.
- Which hidden states the prediction head uses.
- Whether adding labelled examples changes the relevant representation in a useful way.
- Whether the existing supervised training objective could have taught the model to interpret labelled examples.
- Whether there are architectural or context-length limitations.
- How the predicted redshift probability distribution is produced.

Distinguish carefully between architectural feasibility and demonstrated empirical ability.

If direct in-context conditioning is unlikely to work, identify the smallest scientifically defensible modification that would allow us to test the hypothesis.

**Do not immediately launch another QLoRA training run.**

The first objective is to investigate parameter-free target-survey adaptation using an existing checkpoint.

If that proves inadequate, discuss whether episodic SFT or another form of meta-training would be needed to acquire the ability. Treat such training as a separate experimental stage, not as part of the initial calibration procedure.

## 4. Design the calibration experiment

Let a source-survey model have fixed parameters \(\theta\).

A target survey provides a calibration set:

\[
\mathcal C_{\mathrm{target}}
=
\{(\boldsymbol x_i,z_{i,\mathrm{spec}})\}_{i=1}^{N_{\mathrm{cal}}}.
\]

The desired inference is:

\[
p_\theta(z\mid\boldsymbol x_*,\mathcal C_{\mathrm{target}})
\]

with:

\[
\Delta\theta=0.
\]

The scientific question is whether the calibration examples meaningfully improve predictions on independent target-survey galaxies.

Develop and test sensible approaches, potentially including:

- Direct labelled-example conditioning through the existing Qwen input interface.
- Query-dependent selection of relevant calibration galaxies.
- Retrieval-augmented in-context examples.
- Compact summaries of the target calibration distribution.
- Explicitly identified photometric band and survey metadata.
- Other parameter-free conditioning strategies that you consider more appropriate.

These are suggestions, not mandatory implementations. Critically evaluate them.

In particular, distinguish between:

1. A model genuinely using calibration information through its forward pass.
2. An external calibration algorithm correcting the model's outputs.
3. Conventional optimisation or fitting using the calibration set.

The second and third can be useful baselines, but must not be misrepresented as prompt-based or in-context adaptation.

Where possible, retain the full native band information instead of reducing all experiments to common grz bands.

## 5. Experimental comparisons

Start with a small, controlled pilot using an existing Qwen QLoRA checkpoint.

The initial transfer directions are:

- DESI–Legacy → GalaxiesML-Spec
- GalaxiesML-Spec → DESI–Legacy

Begin with one direction if this substantially simplifies debugging.

Explore calibration sample sizes such as:

\[
N_{\rm cal}=0,4,16,64,256
\]

These values are starting points, not fixed requirements.

For large calibration pools, retrieval or summarisation may be preferable to placing every example in the context window.

Evaluate whether the chosen examples are representative of the target photometric population. Example selection must not use the unknown redshift of the query object.

Include strong baselines where feasible:

| Method | Target labels used? | Model parameters updated? |
|---|---|---|
| Direct source-model transfer | No | No |
| In-context calibration | Yes | No |
| Simple global bias correction | Yes | No |
| Local / nearest-neighbour residual correction | Yes | No |
| TabICL with matched labelled context | Yes | No |
| Target-domain QLoRA adaptation | Yes | Yes |

The last baseline is a reference for conventional adaptation rather than part of the parameter-free method.

Evaluate not just absolute performance, but **the additional value provided by the source-trained model**.

For example, determine whether target calibration genuinely transfers learned source information, rather than merely recreating a target-only nearest-neighbour predictor.

A particularly informative control would compare source-trained and otherwise comparable models without source-domain adaptation under the same target calibration budget.

## 6. Scientific evaluation

Use existing evaluation procedures where possible.

The principal metrics should include:

- NMAD.
- Catastrophic-outlier fraction.
- Bias.
- CRPS.
- Cross-entropy or log score.
- PIT and credible-interval coverage.
- Redshift-dependent residuals.
- Tomographic redshift-distribution quality, where supported.

Use the same independent target test objects across comparisons.

The main result should be the relationship between calibration sample size and target-domain performance.

However, **do not optimise only NMAD**.

Previous RLVR experiments demonstrated that point-estimate improvements can accompany degraded probabilistic calibration.

For cosmological applications, the quality of the inferred redshift distribution is scientifically important.

Also investigate whether the calibration produces:

- Only a global redshift offset.
- A colour- or magnitude-dependent correction.
- Better handling of previously unfamiliar bands.
- Improvement in particular redshift regions.
- More appropriately calibrated uncertainty estimates.

Paper I is concerned with cross-survey generalisation within the supported redshift range. Do not turn this experiment into a study of extrapolation beyond the supervised training redshift range.

## 7. Scientific validity and data leakage

Treat data separation as a first-class concern.

Both datasets involve DESI spectroscopy, and GalaxiesML-Spec may contain galaxies also appearing in the DESI–Legacy catalogue.

Before making quantitative cross-survey transfer claims:

- Check relevant object overlap using available identifiers or positional matching where necessary.
- Ensure the calibration set and evaluation queries are disjoint.
- Ensure target-query spectroscopic labels are never exposed through calibration examples, prompts, retrieval databases, or model-selection procedures.
- Check whether any target test objects were previously included in source training under a different photometric representation.
- Separate validation-based method selection from the final test evaluation.

Do not launch an unnecessarily expensive global overlap audit before establishing basic model-interface feasibility. Perform the overlap checks needed for the particular experiment being evaluated.

Keep the photometric feature definitions, uncertainty treatment, and missing-band semantics physically consistent.

A calibration method must not obtain apparent gains merely by exploiting inconsistent data preprocessing or leaking target labels.

## 8. Computational strategy

Be economical with GPU resources.

Start with:

1. Repository and checkpoint inspection.
2. A no-training inference/interface test.
3. A small calibration feasibility pilot.
4. Strong parameter-free baseline comparisons.
5. Larger, repeated evaluation only when justified by the pilot.

Use a limited number of queries and calibration examples for initial screening.

Inspect available hardware and computational costs before launching large-scale experiments.

Do not cancel, overwrite, or interfere with the four existing QLoRA/DoRA experiments.

Put new work in an isolated experiment directory, with reproducible configurations and scripts.

Preserve existing data splits and evaluation code where possible.

Do not introduce unnecessary architectural complexity before testing simpler hypotheses.

## 9. Research independence and critical thinking

You are acting as a research collaborator, not merely implementing a predefined recipe.

Challenge the proposed approach where necessary.

In particular:

- Do not assume a language-model architecture automatically gives useful in-context regression.
- Do not confuse prompt conditioning with parameter updates.
- Do not assume that matching input dimensions implies matching physical features.
- Do not force representations or posteriors from different photometric systems to be identical.
- Do not regard a small improvement over zero-shot transfer as sufficient evidence of scientific novelty.
- Do not attribute improvements to foundation-model pretraining without appropriate controls.
- Do not generalise AION-specific findings to all astronomy-specialised models.

Consider whether this idea is already substantially covered by existing work on tabular in-context learning, few-shot photo-z calibration, test-time adaptation, or domain adaptation.

If literature access is available, review the most relevant work and identify what, if anything, would be novel about our particular approach.

Scientific validity takes priority over obtaining a positive result.

If a simple classical method performs equally well, that is scientifically informative and should be reported.

## 10. Deliverables

Work towards the following outputs.

### A. Feasibility assessment

A concise technical explanation of whether our existing Qwen photo-z implementation can accept and use calibration examples without parameter updates.

Identify the critical architectural constraints and any implementation problems discovered.

### B. Reproducible proof of concept

Implement the minimum necessary code for a genuine parameter-free calibration pilot, if feasible.

Include tests for input consistency, checkpoint equivalence, and calibration/query separation.

### C. Initial experiment

Run a computationally modest experiment comparing direct transfer against calibration-conditioned predictions and basic classical baselines.

Report negative results as clearly as positive ones.

### D. Scientific assessment

Evaluate whether the results support a potentially publishable contribution to:

**Post-training and Cross-survey Generalisation of Foundation Models in Astronomy — I: A Case Study in Photometric Redshift Inference**

Address:

- Whether calibration improves cross-survey performance.
- Whether any improvement comes from useful in-context adaptation.
- How many spectroscopic calibration objects are required.
- Whether source-domain post-training offers advantages beyond simpler approaches.
- Whether the approach appears extensible to additional surveys.
- What would be needed for a stronger Paper I result.

### E. Recommended next experiments

Prioritise follow-up experiments by expected scientific information gained per unit computational cost.

Separate work that is essential for Paper I from more ambitious extensions that could belong in Paper II.

## 11. Execution instructions

Begin by independently examining the actual