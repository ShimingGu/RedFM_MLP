# Experiment A — Warm-start Level-0 QLoRA Control

Implement and execute a controlled continued-SFT experiment using the existing completed GalaxiesML-Spec HSC grizy Qwen3.5-4B-Base QLoRA checkpoint.

## Objective

Establish whether continued training with the **original, unchanged input serializer** preserves or improves the already established photometric-redshift performance.

The archived model achieved approximately:

- Validation NMAD: 0.0268
- Validation CE: 2.4530
- Validation sample: 5,683 objects
- PDF grid: 300 bins, z = 0–4.5

These are reference values, not guaranteed results from a new run.

## Instructions

1. Locate the exact completed HSC QLoRA checkpoint and its matching trained PDF prediction head.
2. Confirm the model is Qwen3.5-4B-Base and inspect the actual adapter configuration.
3. Load **both the trained adapter and trained PDF head**. Do not initialise either from scratch.
4. Use the complete existing HSC training and validation cohorts: 22,444 / 5,683 objects.
5. Retain the original Level-0 input serializer, photometric values, labels, missing-value conventions and PDF grid.
6. Before any optimisation, run a zero-update validation replay and verify that the restored model reproduces the archived predictions and metrics.
7. Continue SFT from that checkpoint, without repeating the three-epoch head-only warmup.

## Matched continued-SFT protocol

For the initial experiment:

- Maximum 2 epochs of additional training.
- No head-only warmup.
- Keep the existing adapter architecture, target modules and rank unchanged.
- Start with adapter LR = 3e-6 and head LR = 1e-5.
- Effective batch size = 16, unless the original training code requires a different interpretation.
- Seed = 1023.
- Evaluate approximately every 250 optimizer updates.
- Select the best checkpoint using source-validation cross-entropy.
- Include the zero-update checkpoint among checkpoint-selection candidates.

These settings must be recorded and used identically for Experiments B and C.

If implementation constraints require changing the protocol, document the change rather than silently substituting another configuration.

## Evaluation

Record:

- Validation CE, NMAD, outlier fraction, bias, CRPS and 68% coverage.
- Training and validation learning curves.
- Performance at zero updates and at the best selected checkpoint.
- Total training updates, wall time and peak GPU memory.
- Adapter and PDF-head parameter changes.

Do not use held-out test objects for model selection.

Once the source checkpoint is selected, run fixed-weight transfer evaluation on the existing audited Legacy and CLAUDS validation queries. Use the same target objects for Experiments A, B and C.

Do not fit or calibrate on target labels.

## Safety and outputs

Do not overwrite the archived model, training results or prior pilot.

Save everything under:

`filter-aware-runs/1023/hsc-warmstart/level0/`

Do not interrupt other GPU jobs. Execute the run when resources are safely available; otherwise prepare the launch command and report that execution remains pending.

Produce a concise report answering:

**Does continued SFT preserve the original HSC capability, and how much do source and cross-survey metrics change without modifying the input representation?**

Implement and run the experiment, rather than stopping after a design proposal.