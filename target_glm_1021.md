# Estimating the practical posttraining limit of the tiny GLM checkpoint

Date: 2026-10-21

**We have not established its posttraining limit.** We established that fixing the implementation makes QLoRA improve slightly. The absolute performance remains poor.

The checkpoint we tested is **GLM-5.2-0.8B-A0.8B**, rather than GLM-5.3. More consequentially, its model card describes a **testing checkpoint trained on a toy copypasta dataset**. Its training history is therefore a major confounder when judging what a 0.8B model can achieve. [Model card](https://huggingface.co/inference-optimization/GLM-5.2-0.8B-A0.8B)

Our adaptation was also restricted: attention/indexer adapters were trainable, while dense MLPs and routed experts remained frozen. The result characterizes that configuration, rather than the whole model's capacity.

## Estimating a practical performance frontier

An exact theoretical ceiling cannot be determined from these runs. We can estimate a **practical performance frontier under a stated training budget**, proceeding in this order:

| Experiment | What it establishes |
|---|---|
| 1. Small-sample overfitting sanity check: fit 512–2,048 training objects and evaluate those same objects | Whether the current GLM pipeline can learn the mapping at all; this does not measure generalization |
| 2. Matched numeric MLP using identical input information, labels, objects, and splits | Whether substantially better predictions are achievable from the same catalogue information |
| 3. Broader QLoRA / final-block fine-tuning (very optional) | Whether adapter placement or restricting trainable weights explains a remaining limitation |

Full-model fine-tuning is reserved for when it is necessary to resolve a specific question left open by these checks; it is not a routine next step. Intermediate-layer and last/mean-pooling comparisons remain optional diagnostics if the earlier checks point to a representation problem.

Full fine-tuning is an empirical reference, **not a guaranteed upper bound**. Research has found gaps between standard low-rank adaptation and full fine-tuning on other tasks; that motivates this comparison but does not predict our astronomy result. [LoRA comparison study](https://arxiv.org/abs/2405.09673)

## Learning curves and interpretation

After the sanity check and matched numeric MLP comparison, measure learning curves if useful, using, for example, **5k, 15k, 30k, and 60k training objects**, keeping validation fixed:

- **Training error falls, validation stalls:** investigate generalization, regularization, and data coverage.
- **Both remain poor:** optimization, representation, initialization, or capacity may be limiting.
- **Broader/full tuning improves substantially:** the current adapter setup was restrictive.
- **Other models perform much better on identical data:** the input information permits better performance.

A fitted learning-curve asymptote can help estimate diminishing returns, but it needs uncertainty estimates and does not prove an irreducible error floor. [Learning-curve methodology](https://arxiv.org/abs/1712.00409)

## Recommended experiment order

The current run commands use **GalaxiesML-Spec**, with native HSC grizy fluxes and errors plus EBV as inputs and **`DESI_fibermap/DESI_redshift` (spectroscopic z) as true z**. Reuse the quality-selected survey cohort and grouped seed-42 splits: 22,444 training, 5,683 validation, and 83,853 held-out test objects, with 300 redshift bins over [0, 4.5]. Neither diagnostic evaluates the held-out test split. Use separate output directories from the completed CLAUDS checks; those earlier results are recorded in [glm52_test_1021.md](glm52_test_1021.md).

1. **Small-sample overfitting sanity check.** Use the current GLM QLoRA setup on 512–2,048 objects and evaluate training fit on those same objects. If it cannot fit this small sample, investigate the pipeline and optimization before scaling up.
2. **Matched numeric MLP.** Compare against GLM using identical input information, labels, objects, and train/validation/test splits. Use validation to assess generalization and select configurations.
3. **Broader QLoRA / final-block fine-tuning (very optional).** Consider these only if the earlier checks leave a useful question about adaptation restrictions. Attempt full-model fine-tuning only if necessary to resolve that question.

For configurations selected for further evaluation, repeat promising settings across three seeds and reserve test evaluation for the final selection. The immediate priorities are the overfitting sanity check followed by the matched numeric MLP; broader tuning is optional follow-up.

The earlier implementation diagnosis and corrected results are recorded in [diagnosis_glm_0906.md](diagnosis_glm_0906.md). The proposed experiments here address remaining performance limitations after that correction.
