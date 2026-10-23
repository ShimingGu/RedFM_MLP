# Have we utilized the potential of Qwen3.5-4B-Base?

Date: 2026-10-22

**The concern is reasonable, but the current results do not establish whether 4B is underused or 0.8B already has enough capacity for this task. Incomplete training and restricted adapter coverage are concrete reasons to withhold judgment.**

## What the current results show

At the latest matched completed epoch examined, the GalaxiesML-Spec QLoRA runs were close:

| After epoch 4 | Validation cross-entropy | Validation NMAD |
|---|---:|---:|
| Qwen3.5-0.8B-Base | 2.8660 | **0.0404** |
| Qwen3.5-4B-Base | **2.8333** | 0.0418 |

These values are a snapshot of the logs inspected for this note, not final results. Both use the same quality-selected cohort: 22,444 training and 5,683 validation objects, with DESI spectroscopic redshift as true z.

The first **three epochs train only the prediction head**, so the epoch-4 comparison includes just **one epoch of adapter training**. The 4B QLoRA run subsequently reached validation CE **2.5228** and NMAD **0.0282 at epoch 7**, with both still improving. The 0.8B run had not reached that stage when inspected; those later 4B values are not a matched-epoch size comparison.

The completed frozen controls show a modest 4B advantage: at checkpoints selected by validation CE, NMAD is **0.04847 for 4B**, versus **0.05105 for 0.8B**. This does not establish either model's adapted performance ceiling.

## A concrete restriction: adapter coverage

The [current launcher](scripts/qwen-survey_posttraining.sh) targets only:

```text
q_proj,k_proj,v_proj,o_proj
```

Inspection of the actual model configurations and module names showed:

| Model | Text blocks | Full-attention blocks receiving adapters | Linear-attention blocks without adapters |
|---|---:|---:|---:|
| 0.8B | 24 | 6 | 18 |
| 4B | 32 | 8 | 24 |

The linear-attention blocks use `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, and `out_proj`, which do not match the configured targets. All MLPs also remain frozen.

All blocks participate in inference and propagate changes from upstream adapters. However, direct adapter updates are confined to the full-attention projections, alongside training the prediction head. This restriction applies to **both sizes**; it is not evidence that broader coverage would uniquely benefit 4B.

The shared protocol additionally uses rank 8, adapter LR `1e-5`, a 256-wide prediction head, and last-token pooling. These are possible constraints to investigate, not established bottlenecks. Input truncation is not indicated by the preparation audits: both models' longest serialized input was 239 tokens, below the 512-token limit.

## Proposed checks, in order

1. **Finish the current runs and compare matched updates.** Examine training loss, validation CE, NMAD, and outlier fraction. Compare convergence and validation-selected checkpoints rather than elapsed wall time or unequal training stages.
2. **Test adapter coverage before increasing rank.** Compare current targets, full plus linear-attention projections, and those projections plus MLPs. Keep the cohort, splits, formatting, initial rank, and other optimization settings fixed. Apply the same comparison to both sizes and verify actual module coverage before launching.
3. **Interpret training and validation together.** If 4B fits training better without improving validation, investigate generalization and data coverage. If broader adaptation substantially improves 4B, the current adaptation scope was restrictive. If both sizes reach similar training and validation performance after suitable adaptation, that supports the possibility that 0.8B is sufficient for the present inputs and sample.

Adapter placement materially affected performance in the original [QLoRA experiments](https://arxiv.org/html/2305.14314). That motivates a coverage ablation; it does not establish the optimum for Qwen3.5 or photo-z learning.

## Interpretation

Similar performance under one shared recipe cannot distinguish **sufficient small-model capacity** from **restricted large-model adaptation**, incomplete optimization, or a representation/interface limitation. A larger model is not guaranteed to improve this numerical mapping merely because it has more parameters.

The immediate evidence supports completing the current comparison and checking adapter coverage. It does not yet justify declaring that the larger model's potential has been exhausted or that full fine-tuning is necessary.

No training configuration was changed and no new training or held-out test evaluation was performed for this note.
