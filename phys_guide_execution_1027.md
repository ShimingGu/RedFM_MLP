# Sequential execution — Stage 0.5 followed by Stage 1

Execute the attached Stage 0.5 (./phys_guide_1027aa.md) and Stage 1 (./phys_guide_1027b.md) prompts sequentially, in that order.

**Stage 0.5 must finish before Stage 1 begins.**

Instructions:

1. Complete all feasible Stage 0.5 inference-only experiments.
2. Save the Stage 0.5 scientific report, metrics, figures, manifests and reproduction commands.
3. Review the findings for implementation errors that would invalidate Stage 1, particularly in photometric units, filter-response integration, tokenizer/pooling behaviour, and data leakage.
4. If there is no blocking correctness problem, proceed automatically to Stage 1 without requesting further confirmation.
5. Use the original Stage 1 prompt without changing its scientific hypotheses or benchmark definitions.
6. Stage 1 must remain independent of Stage 0.5 model predictions: do not tune synthetic SED distributions, redshift labels or filter configurations to maximise agreement with CLAUDS target labels.
7. Perform Stage 1's inference-only probes and small frozen-representation comparisons first. Do not initiate large QLoRA or DoRA jobs.
8. If Stage 0.5 discovers a serious unresolved error that would invalidate Stage 1, fix and validate it if possible. Otherwise document the blocker and stop before invalid experiments.
9. Keep the two experiment directories and reports separate. Preserve all archived checkpoints, validation splits and historical results.
10. Do not interrupt other GPU jobs.

At completion, provide two separate reports and a short joint scientific assessment addressing:

- How much of the current Qwen failure can be associated with input fragility, brightness extrapolation or noise?
- Does pretrained Qwen provide measurable advantages on controlled astrophysical reasoning or cross-filter tasks?
- Is there any evidence that Qwen exploits pretrained astrophysical knowledge rather than only learning a supervised regression mapping?
- Based on the results, which Stage 2 architecture and controls should be prioritised?

**Proceed autonomously through both stages unless a genuine correctness or resource blocker prevents scientifically valid execution.**