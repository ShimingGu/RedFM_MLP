# Estimating the practical posttraining limit of the tiny GLM checkpoint

Date: 2026-10-21

**We have not established its posttraining limit.** We established that fixing the implementation makes QLoRA improve slightly. The absolute performance remains poor.

The checkpoint we tested is **GLM-5.2-0.8B-A0.8B**, rather than GLM-5.3. More consequentially, its model card describes a **testing checkpoint trained on a toy copypasta dataset**. Its training history is therefore a major confounder when judging what a 0.8B model can achieve. [Model card](https://huggingface.co/inference-optimization/GLM-5.2-0.8B-A0.8B)

Our adaptation was also restricted: attention/indexer adapters were trainable, while dense MLPs and routed experts remained frozen. The result characterizes that configuration, rather than the whole model's capacity.

## Estimating a practical performance frontier

An exact theoretical ceiling cannot be determined from these runs. We can estimate a **practical performance frontier under a stated training budget**:

| Experiment | What it establishes |
|---|---|
| Fit 512–2,048 training objects, evaluating those same objects | Whether the pipeline can learn the mapping at all; this does not measure generalization |
| Compare intermediate layers and last/mean pooling | Whether useful information is hidden by the current representation interface |
| Compare current adapters, broader adapters, and higher ranks | Whether adapter placement or expressivity restricts learning |
| Unfreeze the final block, then attempt full-model fine-tuning | Whether restricting trainable weights is the main bottleneck |
| Compare a numeric MLP and Qwen using identical inputs, labels, and splits | Whether substantially better predictions are achievable from the same catalogue information |

Full fine-tuning is an empirical reference, **not a guaranteed upper bound**. Research has found gaps between standard low-rank adaptation and full fine-tuning on other tasks; that motivates this comparison but does not predict our astronomy result. [LoRA comparison study](https://arxiv.org/abs/2405.09673)

## Learning curves and interpretation

Measure learning curves using, for example, **5k, 15k, 30k, and 60k training objects**, keeping validation fixed:

- **Training error falls, validation stalls:** investigate generalization, regularization, and data coverage.
- **Both remain poor:** optimization, representation, initialization, or capacity may be limiting.
- **Broader/full tuning improves substantially:** the current adapter setup was restrictive.
- **Other models perform much better on identical data:** the input information permits better performance.

A fitted learning-curve asymptote can help estimate diminishing returns, but it needs uncertainty estimates and does not prove an irreducible error floor. [Learning-curve methodology](https://arxiv.org/abs/1712.00409)

## Recommended next experiment

Run a small screening experiment: **current QLoRA versus broader adapters versus final-block fine-tuning, alongside a numeric MLP control**. Screen on validation, repeat promising configurations across three seeds, and reserve test evaluation for the final selection.

That would tell us much more than extending the present run or switching adapter algorithms alone.

The earlier implementation diagnosis and corrected results are recorded in [diagnosis_glm_0906.md](diagnosis_glm_0906.md). The proposed experiments here address remaining performance limitations after that correction.
