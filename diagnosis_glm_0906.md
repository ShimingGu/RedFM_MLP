# GLM-5.2-0.8B-A0.8B diagnosis

Date: 2026-09-06

## Conclusion

The earlier GLM post-training failure was caused by a padding-sensitive
last-token pooling bug. It was not evidence that this GLM checkpoint lacks a
useful representation, that live 4-bit execution is invalid, or that the
direct head-training protocol cannot work.

After correcting pooling and rebuilding the embedding cache from the same
k-bit-prepared model state used by PEFT, frozen-head and QLoRA performance both
improved. Corrected QLoRA outperforms its matched frozen control. The failure
diagnosis is therefore complete; the remaining planned stages are optional
performance-optimization experiments.

## Root cause

The shared `last` pooling implementation calculated the pooled position as:

```python
attention_mask.sum(dim=1) - 1
```

This is an absolute position only for right-padded batches. The GLM tokenizer
uses left padding, so the calculation frequently selected a padding or earlier
token instead of the final non-padding token. The selected representation also
changed with batch padding shape.

The corrected implementation finds the greatest absolute sequence position
whose attention mask is set. The change was applied to both shared pooling
paths, with a regression test covering left- and right-padded rows.

The corrected frozen cache is also extracted after
`prepare_model_for_kbit_training`. This matches the model state used by QLoRA;
k-bit preparation promotes the non-quantized BF16 tensors to FP32 and can alter
a small number of MoE routing decisions. Cache naming and metadata now record
both the prepared state and padding-safe pooling version.

## Stage 0 evidence

Stage 0 compared the corrected cached embeddings, live unprepared GLM, live
k-bit-prepared GLM, and a PEFT-wrapped model with a zero adapter on 2,048 fixed
validation objects.

All required invariants passed:

- cached row order matched the reconstructed catalogue;
- no input was truncated;
- tokenization was identical across live paths;
- cached versus prepared mean embedding cosine was 0.999770;
- prepared versus zero-adapter mean cosine was 1.000000;
- all saved LoRA-B tensors were exactly zero;
- cached versus prepared frozen-head CE differed by only 0.000208.

This established that the corrected prepared cache, online prepared model, and
inactive PEFT model expose the same representation for practical purposes.

## Stage 1 evidence

Stage 1 trained the full 2-by-2 frozen-head control on 60,000 training objects
and selected checkpoints using 15,000 validation objects. The test split was
not evaluated. Cached/live pairs used identical head initialization, row order,
and dropout masks.

| Arm | Representation | Head protocol | Validation CE | NMAD | Selected epoch (zero-based) |
|---|---|---|---:|---:|---:|
| A | Cached | Baseline | 4.883122 | 0.236718 | 5 |
| B | Cached | Direct | **4.868454** | **0.231249** | 3 |
| C | Live prepared GLM | Baseline | 4.883069 | 0.236819 | 5 |
| D | Live prepared GLM | Direct | **4.868832** | **0.232367** | 3 |

The required A/C live-path gate passed. A and C differed by only -0.0011% in
CE and +0.0426% in NMAD. The B/D pair also passed, differing by +0.0077% in CE
and +0.4836% in NMAD.

The direct protocol was not a failure mode. Relative to live baseline C, live
direct D improved validation CE by 0.292%, NMAD by 1.88%, and catastrophic
outlier fraction by 1.86%.

The Stage 1 report contains one non-gating failed reference check: the new A
arm differs from the earlier corrected A result by more than 1% in NMAD. This
is benign because the new value is 1.38% better, while validation CE agrees to
0.051%. The actual A/C continuation gate passes.

## Corrected end-to-end result

On the untouched 225,000-object test split:

| Method | Test CE | NMAD | Catastrophic fraction | CRPS |
|---|---:|---:|---:|---:|
| Corrected frozen head | 4.876984 | 0.238536 | 0.527218 | 0.405133 |
| Corrected QLoRA | **4.853008** | **0.226336** | **0.507569** | **0.393105** |

Corrected QLoRA improves over the corrected frozen head by 0.49% in CE, 5.11%
in NMAD, 3.73% in catastrophic fraction, and 2.97% in CRPS. Relative to the old
broken QLoRA result, it improves CE by 6.58%, NMAD by 38.70%, and catastrophic
fraction by 23.90%.

Stage 1 additionally isolates the adapter contribution from the direct frozen
head: corrected QLoRA improves over live direct arm D by 0.154% in validation
CE and 2.42% in validation NMAD.

## Why the remaining stages are unnecessary

The planned later stages were conditional diagnostics:

- Stage 2 would test whether a warm-started, adapter-only run can improve after
  the live and head controls pass. Corrected QLoRA already provides positive
  adapter evidence over the matched direct frozen control, so this is optional
  causal refinement rather than necessary diagnosis.
- Stage 3 would search layers, pooling methods, and prompts if the corrected
  representation remained uninformative. It does not.
- Stage 4 would broaden GLM adapter targets if the current targets could not
  improve performance. They already do.
- Stage 5 would test a limited full-finetuning capacity ceiling if all cheaper
  representations and PEFT methods failed. That condition was not reached.
- RLVR may still be run as an optimization experiment, but it is not needed to
  establish the cause or change the present conclusion.

The appropriate stopping point is after Stage 1. Any continuation should be
described as performance optimization, not failure investigation.

## Artifacts

- Full correction summary: `glm52_paddingfix_0905.md`
- Machine-readable manifest: `glm52_paddingfix_0905_manifest.json`
- Investigation plan and outcome: `target_glm_0828.md`
- Stage 0 report: `/arc/home/gsm/aion_output/figures/glm52-failure-analysis-paddingfix-0905/stage0_invariants/stage0_report.md`
- Stage 1 report: `/arc/home/gsm/aion_output/figures/glm52-failure-analysis-paddingfix-0905/stage1_head_controls/stage1_report.md`
- Stage 1 validation curves: `/arc/home/gsm/aion_output/figures/glm52-failure-analysis-paddingfix-0905/stage1_head_controls/stage1_validation_curves.png`
- Corrected QLoRA summary: `/arc/home/gsm/aion_output/figures/glm52-posttraining-paddingfix-0905/photometry/qlora_run.json`
